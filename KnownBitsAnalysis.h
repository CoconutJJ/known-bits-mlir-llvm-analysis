

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <utility>

struct KnownBits {
        enum BitState { ZERO, ONE, UNKNOWN };

        llvm::SmallVector<BitState, 64> state;
        bool initialized = false;

        KnownBits ()
        {
                state.assign (64, UNKNOWN);
        }

        KnownBits (uint64_t constant) : state (64, UNKNOWN), initialized (true)
        {
                for (int bit = 0; bit < 64; bit++)
                        if ((constant & (1 << bit)) > 0)
                                this->state[bit] = ONE;
                        else
                                this->state[bit] = ZERO;
        }

        static struct KnownBits unknown ()
        {
                struct KnownBits bits;
                bits.initialized = true;
                bits.state.assign (64, UNKNOWN);

                return bits;
        }

        static std::pair<enum BitState, enum BitState> add_bits (enum BitState a, enum BitState b, enum BitState carry)
        {
                switch (a) {
                case ZERO: {
                        switch (b) {
                        case ZERO: {
                                switch (carry) {
                                case ZERO: return {ZERO, ZERO};
                                case ONE: return {ONE, ZERO};
                                case UNKNOWN: return {UNKNOWN, ZERO};
                                }
                                break;
                        }
                        case ONE: {
                                switch (carry) {
                                case ZERO: return {ONE, ZERO};
                                case ONE: return {ZERO, ONE};
                                case UNKNOWN: return {UNKNOWN, UNKNOWN};
                                }
                                break;
                        }
                        case UNKNOWN: {
                                switch (carry) {
                                case ZERO: return {UNKNOWN, ZERO};
                                case ONE: return {UNKNOWN, UNKNOWN};
                                case UNKNOWN: return {UNKNOWN, UNKNOWN};
                                }
                                break;
                        }
                        }
                        break;
                }

                case ONE: {
                        switch (b) {
                        case ZERO: {
                                switch (carry) {
                                case ZERO: return {ONE, ZERO};
                                case ONE: return {ZERO, ONE};
                                case UNKNOWN: return {UNKNOWN, UNKNOWN};
                                }
                                break;
                        }
                        case ONE: {
                                switch (carry) {
                                case ZERO: return {ZERO, ONE};
                                case ONE: return {ONE, ONE};
                                case UNKNOWN: return {UNKNOWN, ONE};
                                }
                                break;
                        }
                        case UNKNOWN: {
                                switch (carry) {
                                case ZERO: return {UNKNOWN, UNKNOWN};
                                case ONE: return {UNKNOWN, ONE};
                                case UNKNOWN: return {UNKNOWN, UNKNOWN};
                                }
                                break;
                        }
                        }
                        break;
                }

                case UNKNOWN: {
                        switch (b) {
                        case ZERO: {
                                switch (carry) {
                                case ZERO: return {UNKNOWN, ZERO};
                                case ONE: return {UNKNOWN, UNKNOWN};
                                case UNKNOWN: return {UNKNOWN, UNKNOWN};
                                }
                                break;
                        }
                        case ONE: {
                                switch (carry) {
                                case ZERO: return {UNKNOWN, UNKNOWN};
                                case ONE: return {UNKNOWN, ONE};
                                case UNKNOWN: return {UNKNOWN, UNKNOWN};
                                }
                                break;
                        }
                        case UNKNOWN: {
                                switch (carry) {
                                case ZERO: return {UNKNOWN, UNKNOWN};
                                case ONE: return {UNKNOWN, UNKNOWN};
                                case UNKNOWN: return {UNKNOWN, UNKNOWN};
                                }
                                break;
                        }
                        }
                        break;
                }
                }

                return {UNKNOWN, UNKNOWN};
        }

        static enum BitState invert (enum BitState bit)
        {
                if (bit == ONE)
                        return ZERO;
                if (bit == ZERO)
                        return ONE;

                return UNKNOWN;
        }

        struct KnownBits add (const struct KnownBits &rhs)
        {
                if (!initialized || !rhs.initialized)
                        return KnownBits ();

                struct KnownBits newState = KnownBits::unknown ();

                enum BitState carry       = ZERO;

                for (int bit = 0; bit < 64; bit++) {
                        std::pair result    = KnownBits::add_bits (this->state[bit], rhs.state[bit], carry);
                        newState.state[bit] = result.first;
                        carry               = result.second;
                }

                return newState;
        }

        struct KnownBits sub (const struct KnownBits &rhs)
        {
                if (!initialized || !rhs.initialized)
                        return KnownBits ();
                struct KnownBits newState = KnownBits::unknown ();

                enum BitState carry       = ONE;

                for (int bit = 0; bit < 64; bit++) {
                        std::pair result =
                                KnownBits::add_bits (this->state[bit], KnownBits::invert (state[bit]), carry);
                        newState.state[bit] = result.first;
                        carry               = result.second;
                }

                return newState;
        }

        KnownBits constantLeftShift (int amount) const
        {
                if (!initialized)
                        return KnownBits ();
                struct KnownBits newState = KnownBits::unknown ();

                for (int bit = 0; bit < 64 - amount; bit++)
                        newState.state[bit + amount] = this->state[bit];

                for (int bit = 0; bit < amount; bit++)
                        newState.state[bit] = ZERO;

                return newState;
        }

        struct KnownBits mul (const struct KnownBits &rhs) const
        {
                if (!initialized || !rhs.initialized)
                        return KnownBits ();
                struct KnownBits result (0);

                for (int bit = 0; bit < 64; bit++)

                        if (rhs.state[bit] == ONE)
                                result = result.add (this->constantLeftShift (bit));
                        else if (rhs.state[bit] == UNKNOWN)
                                result = KnownBits::join (result.add (this->constantLeftShift (bit)), result);
                return result;
        }

        static KnownBits join (const KnownBits &lhs, const KnownBits &rhs)
        {
                if (!lhs.initialized)
                        return rhs;
                if (!rhs.initialized)
                        return lhs;

                struct KnownBits newState = KnownBits::unknown ();

                for (int bit = 0; bit < 64; bit++)

                        if (lhs.state[bit] == rhs.state[bit])
                                newState.state[bit] = lhs.state[bit];
                        else
                                newState.state[bit] = UNKNOWN;

                return newState;
        }

        bool operator== (const KnownBits &rhs) const
        {
                for (int bit = 0; bit < 64; bit++) 
                        if (this->state[bit] != rhs.state[bit])
                                return false;

                return true;
        }
};

using KnownBitsState = mlir::dataflow::Lattice<KnownBits>;

class KnownBitsAnalysis : mlir::dataflow::SparseForwardDataFlowAnalysis<KnownBitsState> {
    public:
        using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

        mlir::LogicalResult visitOperation (mlir::Operation *op,
                                            llvm::ArrayRef<KnownBitsState *> operands,
                                            llvm::ArrayRef<KnownBitsState *> results)
        {
                if (llvm::isa<mlir::LLVM::AddOp> (op)) {
                        struct KnownBits &lhs = operands[0]->getValue ();

                        struct KnownBits &rhs = operands[1]->getValue ();

                        struct KnownBits sum  = lhs.add (rhs);

                        propagateIfChanged (results[0], results[0]->join (sum));
                } else if (llvm::isa<mlir::LLVM::SubOp> (op)) {
                        struct KnownBits &lhs = operands[0]->getValue ();

                        struct KnownBits &rhs = operands[1]->getValue ();

                        struct KnownBits sum  = lhs.sub (rhs);

                        propagateIfChanged (results[0], results[0]->join (sum));
                } else if (llvm::isa<mlir::LLVM::MulOp> (op)) {
                        struct KnownBits &lhs = operands[0]->getValue ();

                        struct KnownBits &rhs = operands[1]->getValue ();

                        struct KnownBits prod = lhs.mul (rhs);

                        propagateIfChanged (results[0], results[0]->join (prod));
                }
        }
};
