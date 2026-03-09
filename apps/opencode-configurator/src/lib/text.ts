export const wrapText = (value: string, width: number) => {
  if (value.length <= width) {
    return [value];
  }

  const words = value.split(" ");
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    if (word.length > width) {
      if (current.length > 0) {
        lines.push(current);
        current = "";
      }

      for (let index = 0; index < word.length; index += width) {
        lines.push(word.slice(index, index + width));
      }

      continue;
    }

    const next = current.length === 0 ? word : `${current} ${word}`;
    if (next.length <= width) {
      current = next;
      continue;
    }

    if (current.length > 0) {
      lines.push(current);
    }

    current = word;
  }

  if (current.length > 0) {
    lines.push(current);
  }

  return lines;
};

export const truncateText = (value: string, width: number) => {
  if (width <= 0) {
    return "";
  }

  if (value.length <= width) {
    return value;
  }

  if (width <= 1) {
    return value.slice(0, width);
  }

  return `${value.slice(0, width - 1)}…`;
};

export const clamp = (value: number, min: number, max: number) => {
  return Math.min(Math.max(value, min), max);
};

export const formatModelRow = (input: { id: string; reasoning: boolean; output: number }) => {
  return `- ${input.id}  ${input.reasoning ? "reasoning" : "plain"}  out ${input.output}`;
};

export const isConfirmKey = (name?: string) => {
  return name === "enter" || name === "return";
};
