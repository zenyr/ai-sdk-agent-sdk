import { z } from "zod";

import {
  isRecord,
  readArray,
  readRecord,
  readString,
} from "../shared/type-readers";

type SupportedJsonType =
  | "string"
  | "number"
  | "integer"
  | "boolean"
  | "null"
  | "array"
  | "object";

const isSupportedJsonType = (value: unknown): value is SupportedJsonType => {
  return (
    value === "string" ||
    value === "number" ||
    value === "integer" ||
    value === "boolean" ||
    value === "null" ||
    value === "array" ||
    value === "object"
  );
};

const readBoolean = (
  record: Record<string, unknown>,
  key: string,
): boolean | undefined => {
  const value = record[key];
  return typeof value === "boolean" ? value : undefined;
};

const readRequiredSet = (schema: Record<string, unknown>): Set<string> => {
  const requiredValues = readArray(schema, "required");
  if (requiredValues === undefined) {
    return new Set<string>();
  }

  const requiredKeys = requiredValues.filter((value) => {
    return typeof value === "string";
  });

  return new Set(requiredKeys);
};

const hasNullInTypeArray = (types: unknown[]): boolean => {
  return types.some((value) => value === "null");
};

const readEnumValues = (schema: Record<string, unknown>): unknown[] | undefined => {
  const enumValues = readArray(schema, "enum");
  if (enumValues === undefined || enumValues.length === 0) {
    return undefined;
  }

  return enumValues;
};

const applySchemaDescription = (
  valueSchema: z.ZodTypeAny,
  schema: Record<string, unknown>,
): z.ZodTypeAny => {
  const description = readString(schema, "description");
  if (description === undefined || description.length === 0) {
    return valueSchema;
  }

  return valueSchema.describe(description);
};

const applyEnumConstraint = (
  valueSchema: z.ZodTypeAny,
  enumValues: unknown[] | undefined,
): z.ZodTypeAny => {
  if (enumValues === undefined) {
    return valueSchema;
  }

  return valueSchema.refine((value) => {
    return enumValues.some((enumValue) => enumValue === value);
  });
};

const buildObjectSchema = (schema: Record<string, unknown>): z.ZodTypeAny => {
  const properties = readRecord(schema, "properties") ?? {};
  const requiredKeys = readRequiredSet(schema);

  const shape: Record<string, z.ZodTypeAny> = {};
  for (const [propertyKey, propertySchema] of Object.entries(properties)) {
    const valueSchema = buildValueSchema(propertySchema);

    shape[propertyKey] = requiredKeys.has(propertyKey)
      ? valueSchema
      : valueSchema.optional();
  }

  let objectSchema: z.ZodTypeAny = z.object(shape);

  const additionalProperties = readBoolean(schema, "additionalProperties");
  if (additionalProperties === false) {
    objectSchema = objectSchema.strict();
  }

  if (additionalProperties !== false) {
    objectSchema = objectSchema.passthrough();
  }

  return objectSchema;
};

const buildArraySchema = (schema: Record<string, unknown>): z.ZodTypeAny => {
  const itemSchema = schema.items;
  return z.array(buildValueSchema(itemSchema));
};

const buildSchemaFromType = (
  schema: Record<string, unknown>,
  jsonType: SupportedJsonType,
): z.ZodTypeAny => {
  if (jsonType === "string") {
    return z.string();
  }

  if (jsonType === "number") {
    return z.number();
  }

  if (jsonType === "integer") {
    return z.number().int();
  }

  if (jsonType === "boolean") {
    return z.boolean();
  }

  if (jsonType === "null") {
    return z.null();
  }

  if (jsonType === "array") {
    return buildArraySchema(schema);
  }

  if (jsonType === "object") {
    return buildObjectSchema(schema);
  }

  return z.any();
};

const buildValueSchema = (schemaValue: unknown): z.ZodTypeAny => {
  if (!isRecord(schemaValue)) {
    return z.any();
  }

  const enumValues = readEnumValues(schemaValue);

  const rawType = schemaValue.type;
  if (Array.isArray(rawType)) {
    const nonNullTypes = rawType.filter((value) => {
      return isSupportedJsonType(value) && value !== "null";
    });

    const selectedType = nonNullTypes[0];
    let valueSchema =
      selectedType === undefined
        ? z.any()
        : buildSchemaFromType(schemaValue, selectedType);

    valueSchema = applyEnumConstraint(valueSchema, enumValues);
    valueSchema = applySchemaDescription(valueSchema, schemaValue);

    if (hasNullInTypeArray(rawType)) {
      valueSchema = valueSchema.nullable();
    }

    return valueSchema;
  }

  if (isSupportedJsonType(rawType)) {
    let valueSchema = buildSchemaFromType(schemaValue, rawType);
    valueSchema = applyEnumConstraint(valueSchema, enumValues);
    valueSchema = applySchemaDescription(valueSchema, schemaValue);

    const nullable = readBoolean(schemaValue, "nullable") === true;
    if (nullable) {
      valueSchema = valueSchema.nullable();
    }

    return valueSchema;
  }

  return applySchemaDescription(z.any(), schemaValue);
};

export const buildZodRawShapeFromToolInputSchema = (
  inputSchema: unknown,
): Record<string, z.ZodTypeAny> => {
  if (!isRecord(inputSchema)) {
    return {};
  }

  const properties = readRecord(inputSchema, "properties");
  if (properties === undefined) {
    return {};
  }

  const requiredKeys = readRequiredSet(inputSchema);
  const shape: Record<string, z.ZodTypeAny> = {};

  for (const [propertyKey, propertySchema] of Object.entries(properties)) {
    const valueSchema = buildValueSchema(propertySchema);

    shape[propertyKey] = requiredKeys.has(propertyKey)
      ? valueSchema
      : valueSchema.optional();
  }

  return shape;
};
