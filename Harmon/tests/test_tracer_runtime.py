"""CPU contracts for the complete TRACER-LoRA runtime loop."""

import math
import sys
import unittest
from pathlib import Path

import torch


HARMON_ROOT = Path(__file__).resolve().parents[1]
if str(HARMON_ROOT) not in sys.path:
    sys.path.insert(0, str(HARMON_ROOT))

from src.models.dllm.guard.guard_lora import RankGate  # noqa: E402
from src.models.dllm.guard.risk_control import (  # noqa: E402
    CommitRevisionDecision,
    CommitRevisionPolicy,
    advance_corruption_state,
    advance_masked_tokens,
    apply_commit_revision,
    compute_trajectory_features,
    transfer_schedule_to_mask_probs,
)


class TracerStateContractTest(unittest.TestCase):
    def test_round_state_is_pre_commit_state_with_forward_delta(self):
        states = transfer_schedule_to_mask_probs([1, 1, 1, 1], 4)
        torch.testing.assert_close(
            states[:, 0], torch.tensor([0.0, 0.25, 0.50, 0.75]))
        expected_u = -torch.log(torch.tensor([1.0, 0.75, 0.50, 0.25]))
        torch.testing.assert_close(states[:, 1], expected_u)
        next_u = -torch.log(torch.tensor([0.75, 0.50, 0.25, 1e-6]))
        torch.testing.assert_close(states[:, 2], next_u - expected_u)

    def test_training_state_advances_by_fixed_schedule_fraction(self):
        state_k = torch.tensor([[0.25, -math.log(0.75), 0.0]])
        state_kp1 = advance_corruption_state(
            state_k, committed_fraction=0.25)
        expected_u = -math.log(0.5)
        torch.testing.assert_close(
            state_kp1,
            torch.tensor([[0.50, expected_u,
                           expected_u - (-math.log(0.75))]]),
        )

    def test_training_tokens_advance_one_commit_per_block(self):
        mask_id = 9
        input_ids = torch.full((1, 8), mask_id, dtype=torch.long)
        response_mask = torch.ones(1, 8, dtype=torch.bool)
        block_indices = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]])
        logits = torch.zeros(1, 8, 10)
        # Candidate id 3 is increasingly confident inside each block.
        logits[:, :, 3] = torch.tensor(
            [[0.1, 0.2, 0.9, 0.3, 0.4, 0.8, 0.2, 0.1]])

        next_ids, commit_mask = advance_masked_tokens(
            input_ids=input_ids,
            logits=logits,
            response_mask=response_mask,
            block_indices=block_indices,
            mask_token_id=mask_id,
            committed_fraction=0.25,
        )

        self.assertEqual(commit_mask.nonzero().tolist(), [[0, 2], [0, 5]])
        self.assertEqual(next_ids[0, 2].item(), 3)
        self.assertEqual(next_ids[0, 5].item(), 3)
        self.assertEqual(int((next_ids == mask_id).sum()), 6)


class TracerRevisionTransitionTest(unittest.TestCase):
    def test_evicted_commit_is_replaced_by_mask_token(self):
        decision = CommitRevisionDecision(
            committed_mask=torch.tensor([[False, True, True, False]]),
            new_commit_mask=torch.tensor([[False, False, True, False]]),
            remask_mask=torch.tensor([[True, False, False, False]]),
            retained_mask=torch.tensor([[False, True, False, False]]),
        )
        next_ids = apply_commit_revision(
            block_ids=torch.tensor([[11, 12, 99, 99]]),
            sampled_ids=torch.tensor([[21, 22, 23, 24]]),
            mask_token_id=99,
            decision=decision,
        )
        torch.testing.assert_close(next_ids, torch.tensor([[99, 12, 23, 99]]))

    def test_policy_consumes_precomputed_shared_risk_without_second_head_call(self):
        hidden = torch.randn(1, 4, 3)
        logits = torch.randn(1, 4, 8)
        features = compute_trajectory_features(
            hidden=hidden,
            logits=logits,
            prev_probs=torch.softmax(torch.randn(1, 4, 8), dim=-1),
            prev_candidates=torch.randint(0, 8, (1, 4)),
            committed_history=torch.tensor([[True, False, False, False]]),
            remask=torch.zeros(1, 4, dtype=torch.bool),
            state=torch.tensor([[0.25, -math.log(0.75), math.log(1.5)]]),
            block_size=4,
        )

        class MustNotRun(torch.nn.Module):
            def forward(self, *args):
                raise AssertionError('RiskHead was executed twice')

        decision = CommitRevisionPolicy(
            block_size=4, max_revision_fraction=0.25
        ).select_committed(
            features=features,
            target_committed_count=2,
            prev_committed=torch.tensor([[True, False, False, False]]),
            risk_head=MustNotRun(),
            risk_scores=torch.tensor([[0.9, 0.1, 0.2, 0.3]]),
        )
        self.assertEqual(int(decision.committed_mask.sum()), 2)
        self.assertTrue(decision.remask_mask[0, 0])


class TracerFactorialSwitchTest(unittest.TestCase):
    def test_first_round_without_lagged_evidence_is_neutral_after_training(self):
        gate = RankGate(rank=4, submodel='R6')
        with torch.no_grad():
            gate.raw_beta.fill_(1.0)
            for mlp in (gate.u_t, gate.u_g, gate.u_tg):
                for parameter in mlp.parameters():
                    parameter.normal_(0.0, 0.5)
        scale = gate(
            torch.tensor([[0.5, math.log(2.0), math.log(2.0)]]),
            None,
            n_tokens=2,
        )
        torch.testing.assert_close(scale, torch.ones_like(scale))

    def test_router_off_returns_unit_scale_with_same_gate_parameters(self):
        gate = RankGate(rank=4, submodel='R6')
        with torch.no_grad():
            gate.raw_beta.fill_(1.0)
            for mlp in (gate.u_t, gate.u_g, gate.u_tg):
                for parameter in mlp.parameters():
                    parameter.normal_(0.0, 0.5)
        parameter_keys = tuple(gate.state_dict())
        scale = gate(
            torch.tensor([[0.5, math.log(2.0), math.log(2.0)]]),
            torch.tensor([[[0.8], [0.2]]]),
            n_tokens=2,
            routing_enabled=False,
        )
        torch.testing.assert_close(scale, torch.ones_like(scale))
        self.assertEqual(tuple(gate.state_dict()), parameter_keys)


if __name__ == '__main__':
    unittest.main()
