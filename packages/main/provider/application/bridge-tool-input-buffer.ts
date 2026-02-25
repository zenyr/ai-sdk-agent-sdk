type PendingBridgeToolInput = {
  toolName: string;
  deltas: string[];
};

export type PendingBridgeToolInputs = Map<string, PendingBridgeToolInput>;

type StartPendingBridgeToolInputArgs = {
  pendingBridgeToolInputs: PendingBridgeToolInputs;
  id: string;
  toolName: string;
};

export const startPendingBridgeToolInput = (args: StartPendingBridgeToolInputArgs): void => {
  args.pendingBridgeToolInputs.set(args.id, {
    toolName: args.toolName,
    deltas: [],
  });
};

type AppendPendingBridgeToolInputDeltaArgs = {
  pendingBridgeToolInputs: PendingBridgeToolInputs;
  id: string;
  delta: string;
};

export const appendPendingBridgeToolInputDelta = (
  args: AppendPendingBridgeToolInputDeltaArgs,
): boolean => {
  const pendingBridgeToolInput = args.pendingBridgeToolInputs.get(args.id);
  if (pendingBridgeToolInput === undefined) {
    return false;
  }

  pendingBridgeToolInput.deltas.push(args.delta);
  return true;
};

type FinishPendingBridgeToolInputArgs = {
  pendingBridgeToolInputs: PendingBridgeToolInputs;
  id: string;
};

type FinishedBridgeToolInput = {
  toolName: string;
  rawInput: string;
};

export const finishPendingBridgeToolInput = (
  args: FinishPendingBridgeToolInputArgs,
): FinishedBridgeToolInput | undefined => {
  const pendingBridgeToolInput = args.pendingBridgeToolInputs.get(args.id);
  if (pendingBridgeToolInput === undefined) {
    return undefined;
  }

  args.pendingBridgeToolInputs.delete(args.id);

  return {
    toolName: pendingBridgeToolInput.toolName,
    rawInput: pendingBridgeToolInput.deltas.join(""),
  };
};
