#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace {

constexpr char kRuntimeCallbackTarget[] = "exla_runtime_callback";

// Partitions an io_call as an identity operation on its tensor operand. The
// callback server pid is replicated, while the tensor and aliased result keep
// the sharding selected for the surrounding computation. Each partition then
// invokes the callback handler with its local physical buffer.
class RuntimeCallbackPartitioner final : public xla::CustomCallPartitioner {
 public:
  bool IsCustomCallShardable(
      const xla::HloInstruction* instruction) const override {
    return instruction->operand_count() == 2 &&
           !instruction->shape().IsTuple();
  }

  std::optional<xla::HloSharding> InferShardingFromOperands(
      const xla::HloInstruction* instruction) const override {
    if (!IsCustomCallShardable(instruction) ||
        !instruction->operand(1)->has_sharding()) {
      return std::nullopt;
    }

    return instruction->operand(1)->sharding();
  }

  bool CanPropagateShardingToOperands(
      const xla::HloInstruction*) const override {
    // Operand zero is the replicated callback server pid. Shardy's explicit
    // identity rule propagates the result sharding only to operand one.
    return false;
  }

  bool CanSideEffectingHaveReplicatedSharding() const override { return true; }

  absl::Status Partition(
      xla::spmd::SpmdPartitioningVisitor* partitioner,
      xla::HloInstruction* hlo) const override {
    if (!IsCustomCallShardable(hlo)) {
      return absl::InvalidArgumentError(
          "per-partition EXLA io_call expects one tensor operand and one "
          "tensor result");
    }

    auto callback_pid =
        partitioner->GetPartitionedHlo(hlo->operand(0)).Replicate();
    auto data = partitioner->GetPartitionedHlo(hlo->operand(1));

    xla::HloInstruction* local_call =
        partitioner->builder()->AddInstruction(
            hlo->CloneWithNewOperands(
                data.hlo()->shape(), {callback_pid.hlo(), data.hlo()}));
    local_call->set_sharding(data.sharding());

    xla::spmd::PartitionedHlo result(local_call, hlo->shape(),
                                     partitioner->MakePartitioningState());
    partitioner->SetPartitionedHlo(hlo, result.Reshard(hlo->sharding()));
    return absl::OkStatus();
  }
};

const bool kRuntimeCallbackPartitionerRegistered = [] {
  xla::RegisterCustomCallPartitioner(
      kRuntimeCallbackTarget, std::make_unique<RuntimeCallbackPartitioner>());
  return true;
}();

}  // namespace
