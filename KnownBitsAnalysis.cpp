#include "KnownBitsAnalysis.hpp"

#include <cstdint>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

KnownBits::KnownBits ()
{
        state.assign (64, UNKNOWN);
}

KnownBits::KnownBits (uint64_t constant) : state (64, UNKNOWN), initialized (true)
{
        for (int bit = 0; bit < 64; bit++)
                if ((constant & (uint64_t{1} << bit)) > 0)
                        this->state[bit] = ONE;
                else
                        this->state[bit] = ZERO;
}

KnownBits KnownBits::unknown ()
{
        KnownBits bits;
        bits.initialized = true;
        bits.state.assign (64, UNKNOWN);

        return bits;
}

std::pair<KnownBits::BitState, KnownBits::BitState> KnownBits::add_bits (BitState a, BitState b, BitState carry)
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

KnownBits::BitState KnownBits::invert (BitState bit)
{
        if (bit == ONE)
                return ZERO;
        if (bit == ZERO)
                return ONE;

        return UNKNOWN;
}

KnownBits KnownBits::add (const KnownBits &rhs) const
{
        if (!initialized || !rhs.initialized)
                return KnownBits ();

        KnownBits newState = KnownBits::unknown ();
        BitState carry     = ZERO;

        for (int bit = 0; bit < 64; bit++) {
                std::pair result    = KnownBits::add_bits (this->state[bit], rhs.state[bit], carry);
                newState.state[bit] = result.first;
                carry               = result.second;
        }

        return newState;
}

KnownBits KnownBits::sub (const KnownBits &rhs) const
{
        if (!initialized || !rhs.initialized)
                return KnownBits ();

        KnownBits newState = KnownBits::unknown ();
        BitState carry     = ONE;

        for (int bit = 0; bit < 64; bit++) {
                std::pair result    = KnownBits::add_bits (this->state[bit], KnownBits::invert (rhs.state[bit]), carry);
                newState.state[bit] = result.first;
                carry               = result.second;
        }

        return newState;
}

KnownBits KnownBits::constantLeftShift (int amount) const
{
        if (!initialized)
                return KnownBits ();

        KnownBits newState = KnownBits::unknown ();

        for (int bit = 0; bit < 64 - amount; bit++)
                newState.state[bit + amount] = this->state[bit];

        for (int bit = 0; bit < amount; bit++)
                newState.state[bit] = ZERO;

        return newState;
}

KnownBits KnownBits::mul (const KnownBits &rhs) const
{
        if (!initialized || !rhs.initialized)
                return KnownBits ();

        KnownBits result (0);

        for (int bit = 0; bit < 64; bit++)
                if (rhs.state[bit] == ONE)
                        result = result.add (this->constantLeftShift (bit));
                else if (rhs.state[bit] == UNKNOWN)
                        result = KnownBits::join (result.add (this->constantLeftShift (bit)), result);

        return result;
}

KnownBits KnownBits::join (const KnownBits &lhs, const KnownBits &rhs)
{
        if (!lhs.initialized)
                return rhs;
        if (!rhs.initialized)
                return lhs;

        KnownBits newState = KnownBits::unknown ();

        for (int bit = 0; bit < 64; bit++)
                if (lhs.state[bit] == rhs.state[bit])
                        newState.state[bit] = lhs.state[bit];
                else
                        newState.state[bit] = UNKNOWN;

        return newState;
}

bool KnownBits::operator== (const KnownBits &rhs) const
{
        // An uninitialized lattice element is the optimistic bottom state;
        // initialized all-unknown bits are the conservative entry state.
        // They have the same bit vector but are not semantically equal.
        if (initialized != rhs.initialized)
                return false;

        for (int bit = 0; bit < 64; bit++)
                if (this->state[bit] != rhs.state[bit])
                        return false;

        return true;
}

void KnownBitsAnalysis::setToEntryState (KnownBitsState *lattice)
{
        propagateIfChanged (lattice, lattice->join (KnownBits::unknown ()));
}

void KnownBits::print (llvm::raw_ostream &os) const
{
        if (!initialized) {
                os << "<uninitialized>";
                return;
        }

        os << "0b";

        for (int bit = 63; bit >= 0; --bit) {
                switch (state[bit]) {
                case ZERO: os << '0'; break;
                case ONE: os << '1'; break;
                case UNKNOWN: os << '?'; break;
                }
        }
}

mlir::LogicalResult KnownBitsAnalysis::visitOperation (mlir::Operation *op,
                                                       llvm::ArrayRef<const KnownBitsState *> operands,
                                                       llvm::ArrayRef<KnownBitsState *> results)
{
        if (auto constantOp = llvm::dyn_cast<mlir::LLVM::ConstantOp> (op)) {
                auto intAttr = llvm::dyn_cast<mlir::IntegerAttr> (constantOp.getValue ());

                if (!intAttr || intAttr.getValue ().getBitWidth () > 64) {
                        setToEntryState (results[0]);
                        return mlir::success ();
                }

                uint64_t value = intAttr.getValue ().getZExtValue ();

                propagateIfChanged (results[0], results[0]->join (KnownBits (value)));

                return mlir::success ();
        } else if (llvm::isa<mlir::LLVM::AddOp> (op)) {
                const KnownBits &lhs = operands[0]->getValue ();
                const KnownBits &rhs = operands[1]->getValue ();
                KnownBits sum        = lhs.add (rhs);

                propagateIfChanged (results[0], results[0]->join (sum));
                return mlir::success ();
        } else if (llvm::isa<mlir::LLVM::SubOp> (op)) {
                const KnownBits &lhs = operands[0]->getValue ();
                const KnownBits &rhs = operands[1]->getValue ();
                KnownBits sum        = lhs.sub (rhs);

                propagateIfChanged (results[0], results[0]->join (sum));
                return mlir::success ();
        } else if (llvm::isa<mlir::LLVM::MulOp> (op)) {
                const KnownBits &lhs = operands[0]->getValue ();
                const KnownBits &rhs = operands[1]->getValue ();
                KnownBits prod       = lhs.mul (rhs);

                propagateIfChanged (results[0], results[0]->join (prod));
                return mlir::success ();
        }

        setAllToEntryStates (results);
        return mlir::success ();
}
