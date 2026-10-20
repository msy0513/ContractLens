#!/usr/bin/env python3
"""Bounded symbolic reimplementation of the Honeytoken-Detector rules.

The 2023 paper does not publish its modified HoneyBadger engine.  This module
keeps the paper-referenced HoneyBadger checkout as provenance, while providing
a Python 3 / modern-EVM compatible CFG and symbolic data-flow implementation.
It intentionally emits compact rule flags and audit data rather than traces.
"""

from __future__ import annotations

import dataclasses
import fractions
import time
from collections import defaultdict, deque
from typing import Iterable

import z3


UINT256 = 2**256
TRANSFER = 0xA9059CBB
TRANSFER_FROM = 0x23B872DD


OPCODES = {
    0x00: "STOP", 0x01: "ADD", 0x02: "MUL", 0x03: "SUB", 0x04: "DIV",
    0x05: "SDIV", 0x06: "MOD", 0x07: "SMOD", 0x08: "ADDMOD",
    0x09: "MULMOD", 0x0A: "EXP", 0x0B: "SIGNEXTEND", 0x10: "LT",
    0x11: "GT", 0x12: "SLT", 0x13: "SGT", 0x14: "EQ", 0x15: "ISZERO",
    0x16: "AND", 0x17: "OR", 0x18: "XOR", 0x19: "NOT", 0x1A: "BYTE",
    0x1B: "SHL", 0x1C: "SHR", 0x1D: "SAR", 0x20: "SHA3",
    0x30: "ADDRESS", 0x31: "BALANCE", 0x32: "ORIGIN", 0x33: "CALLER",
    0x34: "CALLVALUE", 0x35: "CALLDATALOAD", 0x36: "CALLDATASIZE",
    0x37: "CALLDATACOPY", 0x38: "CODESIZE", 0x39: "CODECOPY",
    0x3A: "GASPRICE", 0x3B: "EXTCODESIZE", 0x3C: "EXTCODECOPY",
    0x3D: "RETURNDATASIZE", 0x3E: "RETURNDATACOPY", 0x3F: "EXTCODEHASH",
    0x40: "BLOCKHASH", 0x41: "COINBASE", 0x42: "TIMESTAMP", 0x43: "NUMBER",
    0x44: "PREVRANDAO", 0x45: "GASLIMIT", 0x46: "CHAINID",
    0x47: "SELFBALANCE", 0x48: "BASEFEE", 0x49: "BLOBHASH",
    0x4A: "BLOBBASEFEE", 0x50: "POP", 0x51: "MLOAD", 0x52: "MSTORE",
    0x53: "MSTORE8", 0x54: "SLOAD", 0x55: "SSTORE", 0x56: "JUMP",
    0x57: "JUMPI", 0x58: "PC", 0x59: "MSIZE", 0x5A: "GAS",
    0x5B: "JUMPDEST", 0x5F: "PUSH0", 0xA0: "LOG0", 0xA1: "LOG1",
    0xA2: "LOG2", 0xA3: "LOG3", 0xA4: "LOG4", 0xF0: "CREATE",
    0xF1: "CALL", 0xF2: "CALLCODE", 0xF3: "RETURN", 0xF4: "DELEGATECALL",
    0xF5: "CREATE2", 0xFA: "STATICCALL", 0xFD: "REVERT",
    0xFE: "INVALID", 0xFF: "SELFDESTRUCT",
}


@dataclasses.dataclass(frozen=True)
class Instruction:
    pc: int
    opcode: int
    name: str
    argument: int | None = None


@dataclasses.dataclass
class Value:
    expr: z3.BitVecRef
    taints: frozenset[str] = frozenset()
    constant: int | None = None
    coeff: dict[int, fractions.Fraction] = dataclasses.field(default_factory=dict)

    @classmethod
    def const(cls, value: int) -> "Value":
        value %= UINT256
        return cls(z3.BitVecVal(value, 256), constant=value)


@dataclasses.dataclass
class PathSummary:
    selector: int | None
    sstores: list[Value]
    restricted: bool
    log3: bool
    delegatecall: bool


@dataclasses.dataclass
class State:
    index: int
    stack: list[Value]
    memory: dict[int, Value]
    storage: dict[str, Value]
    conditions: list[z3.BoolRef]
    sstores: list[Value]
    restricted: bool = False
    log3: bool = False
    delegatecall: bool = False
    steps: int = 0

    def fork(self, index: int) -> "State":
        return State(index, list(self.stack), dict(self.memory), dict(self.storage),
                     list(self.conditions), list(self.sstores), self.restricted,
                     self.log3, self.delegatecall, self.steps)


def disassemble(bytecode: str) -> list[Instruction]:
    raw = str(bytecode or "").removeprefix("0x")
    if len(raw) % 2:
        raw = "0" + raw
    code = bytes.fromhex(raw) if raw else b""
    result: list[Instruction] = []
    pc = 0
    while pc < len(code):
        opcode = code[pc]
        start = pc
        pc += 1
        if 0x60 <= opcode <= 0x7F:
            width = opcode - 0x5F
            payload = code[pc:pc + width]
            pc += width
            result.append(Instruction(start, opcode, f"PUSH{width}", int.from_bytes(payload, "big")))
        elif 0x80 <= opcode <= 0x8F:
            result.append(Instruction(start, opcode, f"DUP{opcode - 0x7F}"))
        elif 0x90 <= opcode <= 0x9F:
            result.append(Instruction(start, opcode, f"SWAP{opcode - 0x8F}"))
        else:
            result.append(Instruction(start, opcode, OPCODES.get(opcode, f"UNKNOWN_{opcode:02x}")))
    return result


def selector_entries(instructions: list[Instruction]) -> dict[int, int]:
    """Recover Solidity dispatcher PUSH4/EQ/PUSH/JUMPI destinations."""
    result: dict[int, int] = {}
    for i, ins in enumerate(instructions):
        if ins.name != "PUSH4":
            continue
        for j in range(i + 1, min(i + 9, len(instructions))):
            if instructions[j].name != "EQ":
                continue
            destination = None
            for candidate in instructions[j + 1:min(j + 6, len(instructions))]:
                if destination is None and candidate.name.startswith("PUSH") and candidate.argument is not None:
                    destination = int(candidate.argument)
                if candidate.name == "JUMPI":
                    break
            else:
                continue
            if destination is not None:
                result[int(ins.argument)] = destination
                break
    return result


def _merge_taints(*values: Value) -> frozenset[str]:
    return frozenset().union(*(value.taints for value in values))


def _coeff_add(a: dict[int, fractions.Fraction], b: dict[int, fractions.Fraction], sign=1):
    result = defaultdict(fractions.Fraction)
    for key, value in a.items(): result[key] += value
    for key, value in b.items(): result[key] += sign * value
    return {key: value for key, value in result.items() if value}


def _binary(name: str, left: Value, right: Value) -> Value:
    # EVM pops the first argument from the top of stack.  For SUB/DIV, the
    # operation is second-popped op first-popped, represented here as right op left.
    constant = None
    coeff: dict[int, fractions.Fraction] = {}
    if name == "ADD":
        expr, coeff = right.expr + left.expr, _coeff_add(right.coeff, left.coeff)
        if right.constant is not None and left.constant is not None: constant = (right.constant + left.constant) % UINT256
    elif name == "SUB":
        expr, coeff = right.expr - left.expr, _coeff_add(right.coeff, left.coeff, -1)
        if right.constant is not None and left.constant is not None: constant = (right.constant - left.constant) % UINT256
    elif name == "MUL":
        expr = right.expr * left.expr
        if right.constant is not None:
            coeff = {k: v * right.constant for k, v in left.coeff.items()}
        elif left.constant is not None:
            coeff = {k: v * left.constant for k, v in right.coeff.items()}
        if right.constant is not None and left.constant is not None: constant = (right.constant * left.constant) % UINT256
    elif name in {"DIV", "SDIV"}:
        expr = z3.UDiv(right.expr, left.expr)
        if left.constant not in {None, 0}:
            coeff = {k: v / left.constant for k, v in right.coeff.items()}
        if right.constant is not None and left.constant not in {None, 0}: constant = right.constant // left.constant
    elif name == "MOD": expr = z3.URem(right.expr, left.expr)
    elif name == "AND": expr = right.expr & left.expr
    elif name == "OR": expr = right.expr | left.expr
    elif name == "XOR": expr = right.expr ^ left.expr
    elif name == "SHL": expr = right.expr << left.expr
    elif name == "SHR": expr = z3.LShR(right.expr, left.expr)
    elif name == "EQ": expr = z3.If(right.expr == left.expr, z3.BitVecVal(1, 256), z3.BitVecVal(0, 256))
    elif name == "LT": expr = z3.If(z3.ULT(right.expr, left.expr), z3.BitVecVal(1, 256), z3.BitVecVal(0, 256))
    elif name == "GT": expr = z3.If(z3.UGT(right.expr, left.expr), z3.BitVecVal(1, 256), z3.BitVecVal(0, 256))
    elif name == "SLT": expr = z3.If(right.expr < left.expr, z3.BitVecVal(1, 256), z3.BitVecVal(0, 256))
    elif name == "SGT": expr = z3.If(right.expr > left.expr, z3.BitVecVal(1, 256), z3.BitVecVal(0, 256))
    else: expr = z3.BitVec(f"{name}_{id(left)}_{id(right)}", 256)
    return Value(z3.simplify(expr), _merge_taints(left, right), constant, coeff)


def _pop(state: State, name: str) -> Value:
    if state.stack:
        return state.stack.pop()
    return Value(z3.BitVec(f"underflow_{name}_{state.index}_{len(state.stack)}", 256), frozenset({"unknown"}))


def _feasible(conditions: Iterable[z3.BoolRef]) -> bool:
    solver = z3.Solver()
    solver.set(timeout=5)
    solver.add(*conditions)
    return solver.check() != z3.unsat


def _reverse_valid_index(pc_to_index: dict[int, int], destination: Value) -> int | None:
    return pc_to_index.get(destination.constant) if destination.constant is not None else None


def execute_function(instructions: list[Instruction], start_pc: int, selector: int | None,
                     deadline: float, state_limit: int = 64, step_limit: int = 2000) -> list[PathSummary]:
    pc_to_index = {ins.pc: i for i, ins in enumerate(instructions)}
    if start_pc not in pc_to_index:
        return []
    queue = deque([State(pc_to_index[start_pc], [], {}, {}, [], [], steps=0)])
    summaries: list[PathSummary] = []
    seen: dict[tuple[int, int], int] = defaultdict(int)
    processed = 0
    while queue and processed < state_limit and time.monotonic() < deadline:
        state = queue.pop()
        processed += 1
        while 0 <= state.index < len(instructions) and state.steps < step_limit and time.monotonic() < deadline:
            ins = instructions[state.index]
            state.steps += 1
            state.index += 1
            visit = (ins.pc, len(state.stack))
            seen[visit] += 1
            if seen[visit] > 24:
                break
            name = ins.name
            if name.startswith("PUSH"):
                state.stack.append(Value.const(ins.argument or 0)); continue
            if name.startswith("DUP"):
                n = int(name[3:]); state.stack.append(state.stack[-n] if len(state.stack) >= n else _pop(state, name)); continue
            if name.startswith("SWAP"):
                n = int(name[4:])
                if len(state.stack) > n: state.stack[-1], state.stack[-1 - n] = state.stack[-1 - n], state.stack[-1]
                continue
            if name == "POP": _pop(state, name); continue
            if name in {"ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "AND", "OR", "XOR", "SHL", "SHR", "EQ", "LT", "GT", "SLT", "SGT"}:
                state.stack.append(_binary(name, _pop(state, name), _pop(state, name))); continue
            if name == "ISZERO":
                value = _pop(state, name)
                state.stack.append(Value(z3.If(value.expr == 0, z3.BitVecVal(1, 256), z3.BitVecVal(0, 256)), value.taints)); continue
            if name == "NOT":
                value = _pop(state, name); state.stack.append(Value(~value.expr, value.taints)); continue
            if name in {"CALLER", "ORIGIN"}:
                state.stack.append(Value(z3.BitVec(name.lower(), 256), frozenset({"caller"}))); continue
            if name == "CALLVALUE": state.stack.append(Value(z3.BitVec("callvalue", 256), frozenset({"callvalue"}))); continue
            if name == "CALLDATALOAD":
                offset = _pop(state, name)
                arg = (offset.constant - 4) // 32 if offset.constant is not None and offset.constant >= 4 else -1
                coeff = {arg: fractions.Fraction(1)} if arg >= 0 else {}
                taints = {"calldata"}
                if arg >= 0: taints.add(f"arg_{arg}")
                state.stack.append(Value(z3.BitVec(f"calldata_{offset.constant}_{selector}", 256), frozenset(taints), coeff=coeff)); continue
            if name in {"CALLDATASIZE", "CODESIZE", "RETURNDATASIZE", "MSIZE", "GAS", "GASPRICE", "TIMESTAMP", "NUMBER", "CHAINID", "BASEFEE", "ADDRESS", "SELFBALANCE"}:
                state.stack.append(Value(z3.BitVec(f"{name.lower()}_{ins.pc}", 256), frozenset({name.lower()}))); continue
            if name == "MSTORE":
                offset, value = _pop(state, name), _pop(state, name)
                if offset.constant is not None: state.memory[offset.constant] = value
                continue
            if name == "MSTORE8": _pop(state, name); _pop(state, name); continue
            if name == "MLOAD":
                offset = _pop(state, name)
                state.stack.append(state.memory.get(offset.constant, Value(z3.BitVec(f"memory_{offset.constant}_{ins.pc}", 256), frozenset({"memory"})))); continue
            if name == "SHA3":
                offset, size = _pop(state, name), _pop(state, name)
                values = [v for key, v in state.memory.items() if offset.constant is None or size.constant is None or offset.constant <= key < offset.constant + size.constant]
                taints = frozenset({"storage_key"}).union(*(v.taints for v in values))
                state.stack.append(Value(z3.BitVec(f"sha3_{ins.pc}_{len(values)}", 256), taints)); continue
            if name == "SLOAD":
                key = _pop(state, name); token = str(z3.simplify(key.expr))
                value = state.storage.get(token, Value(
                    z3.BitVec(f"storage_{abs(hash(token))}_{ins.pc}", 256),
                    frozenset({"storage"}).union(key.taints),
                ))
                state.stack.append(value); continue
            if name == "SSTORE":
                key, value = _pop(state, name), _pop(state, name)
                state.storage[str(z3.simplify(key.expr))] = value
                state.sstores.append(value); continue
            if name == "JUMP":
                target = _reverse_valid_index(pc_to_index, _pop(state, name))
                if target is None: break
                state.index = target; continue
            if name == "JUMPI":
                destination, condition = _pop(state, name), _pop(state, name)
                taints = condition.taints
                amount_arg = 1 if selector == TRANSFER else 2 if selector == TRANSFER_FROM else None
                amount_taint = f"arg_{amount_arg}" if amount_arg is not None else None
                storage_non_amount_guard = "storage" in taints and (amount_taint is None or amount_taint not in taints)
                if ("caller" in taints and "storage" in taints) or storage_non_amount_guard:
                    state.restricted = True
                target = _reverse_valid_index(pc_to_index, destination)
                if target is not None:
                    branch = state.fork(target)
                    branch.conditions.append(condition.expr != 0)
                    if _feasible(branch.conditions): queue.append(branch)
                state.conditions.append(condition.expr == 0)
                if not _feasible(state.conditions): break
                continue
            if name.startswith("LOG"):
                topics = int(name[3:]); _pop(state, name); _pop(state, name)
                for _ in range(topics): _pop(state, name)
                if name == "LOG3": state.log3 = True
                continue
            if name in {"CALL", "CALLCODE"}:
                for _ in range(7): _pop(state, name)
                state.stack.append(Value(z3.BitVec(f"call_result_{ins.pc}", 256))); continue
            if name in {"DELEGATECALL", "STATICCALL"}:
                for _ in range(6): _pop(state, name)
                if name == "DELEGATECALL": state.delegatecall = True
                state.stack.append(Value(z3.BitVec(f"call_result_{ins.pc}", 256))); continue
            if name in {"RETURN", "REVERT"}: _pop(state, name); _pop(state, name)
            if name in {"STOP", "RETURN", "REVERT", "INVALID", "SELFDESTRUCT"}: break
            # Unknown environment/copy/create opcodes receive conservative stack handling.
            if name in {"BALANCE", "EXTCODESIZE", "EXTCODEHASH", "BLOCKHASH"}:
                _pop(state, name); state.stack.append(Value(z3.BitVec(f"{name}_{ins.pc}", 256), frozenset({"unknown"})))
        summaries.append(PathSummary(selector, state.sstores, state.restricted, state.log3, state.delegatecall))
    return summaries


def _amount_coefficients(summary: PathSummary, selector: int) -> tuple[list[fractions.Fraction], list[fractions.Fraction]]:
    arg = 1 if selector == TRANSFER else 2
    values = [event.coeff.get(arg, fractions.Fraction(0)) for event in summary.sstores]
    return [v for v in values if v > 0], [abs(v) for v in values if v < 0]


def analyze_bytecode(bytecode: str, timeout_seconds: float = 20.0) -> dict:
    started = time.monotonic()
    deadline = started + timeout_seconds
    instructions = disassemble(bytecode)
    entries = selector_entries(instructions)
    summaries: list[PathSummary] = []
    # Analyze all recovered functions, capped deterministically for pathological dispatchers.
    for selector, pc in sorted(entries.items())[:96]:
        if time.monotonic() >= deadline: break
        summaries.extend(execute_function(instructions, pc, selector, deadline))

    transfer_paths = [p for p in summaries if p.selector in {TRANSFER, TRANSFER_FROM}]
    non_transfer = [p for p in summaries if p.selector not in {TRANSFER, TRANSFER_FROM}]
    inconsistent = False
    for path in transfer_paths:
        positives, negatives = _amount_coefficients(path, int(path.selector))
        if positives and negatives and max(positives) != max(negatives):
            inconsistent = True
    restriction = any(path.restricted for path in transfer_paths)
    fake_logs = any(path.log3 and not path.sstores for path in non_transfer)
    minting = any(path.restricted and path.log3 and any(v > 0 for event in path.sstores for v in event.coeff.values()) for path in non_transfer)
    balance_manipulation = any(path.restricted and not path.log3 and any(v != 0 for event in path.sstores for v in event.coeff.values()) for path in non_transfer)
    proxy = any(ins.name == "DELEGATECALL" for ins in instructions)
    elapsed = time.monotonic() - started
    return {
        "transfer_quantity_inconsistency": bool(inconsistent),
        "transfer_restriction": bool(restriction),
        "fake_logs": bool(fake_logs),
        "infinite_minting": bool(minting),
        "balance_manipulation": bool(balance_manipulation),
        "proxy_token": bool(proxy),
        "status": "timeout" if elapsed >= timeout_seconds else "ok",
        "elapsed_seconds": elapsed,
        "instruction_count": len(instructions),
        "selector_count": len(entries),
        "path_count": len(summaries),
        "engine": "bounded_cfg_z3_reimplementation",
    }
