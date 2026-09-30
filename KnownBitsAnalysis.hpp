

#ifndef KNOWN_BITS_ANALYSIS_HPP
#define KNOWN_BITS_ANALYSIS_HPP

#include <cstdint>
#include <utility>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/IR/Operation.h>

struct KnownBits {
        enum BitState { ZERO, ONE, UNKNOWN };

        llvm::SmallVector<BitState, 64> state;
        bool initialized = false;

        KnownBits ();
        KnownBits (uint64_t constant);

        static KnownBits unknown ();
        static std::pair<BitState, BitState> add_bits (BitState a, BitState b, BitState carry);
        static BitState invert (BitState bit);

        KnownBits add (const KnownBits &rhs) const;
        KnownBits sub (const KnownBits &rhs) const;
        KnownBits constantLeftShift (int amount) const;
        KnownBits mul (const KnownBits &rhs) const;

        void print (llvm::raw_ostream &os) const;
        static KnownBits join (const KnownBits &lhs, const KnownBits &rhs);

        bool operator== (const KnownBits &rhs) const;
};

using KnownBitsState = mlir::dataflow::Lattice<KnownBits>;

class KnownBitsAnalysis : public mlir::dataflow::SparseForwardDataFlowAnalysis<KnownBitsState> {
    public:
        using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

        void setToEntryState (KnownBitsState *lattice) override;

        mlir::LogicalResult visitOperation (mlir::Operation *op,
                                            llvm::ArrayRef<const KnownBitsState *> operands,
                                            llvm::ArrayRef<KnownBitsState *> results) override;
};

#endif
