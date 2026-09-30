// Small, self-contained programs for the `known-bits` analysis.
//
// The analysis models values as 64 bits, so all examples use i64.  The
// comments state the expected mathematical value or bit property; the
// accompanying .results file contains the pass's actual annotations.
module {
  // Fully known arithmetic exercises the constant, add, subtract, and
  // multiply transfer functions.
  llvm.func @constant_arithmetic() -> i64 {
    %forty_two = llvm.mlir.constant(42 : i64) : i64
    %thirteen = llvm.mlir.constant(13 : i64) : i64
    %sum = llvm.add %forty_two, %thirteen : i64       // 55
    %difference = llvm.sub %forty_two, %thirteen : i64 // 29
    %product = llvm.mul %forty_two, %thirteen : i64    // 546
    %combined = llvm.add %sum, %product : i64          // 601
    llvm.return %combined : i64
  }

  // Unknown operands stay unknown through addition and subtraction.  A known
  // power-of-two multiplier nevertheless proves the three low bits are zero.
  llvm.func @unknown_input(%x: i64) -> i64 {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %eight = llvm.mlir.constant(8 : i64) : i64
    %plus_zero = llvm.add %x, %zero : i64              // unknown
    %minus_self = llvm.sub %x, %x : i64                // unknown (conservative)
    %times_eight = llvm.mul %x, %eight : i64           // ???...???000
    llvm.return %times_eight : i64
  }

  // Multiplication by zero is exact even when the other operand is unknown.
  llvm.func @annihilator(%x: i64) -> i64 {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %result = llvm.mul %x, %zero : i64                 // 0
    llvm.return %result : i64
  }
}
