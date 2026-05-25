import math
import unittest

import torch

from train.schedulers import (
    EpochWarmupCosineScheduler,
    SCHEDULER_WARM_RESTART,
    SCHEDULER_WARMUP_COSINE,
    create_scheduler,
)


class EpochWarmupCosineSchedulerTest(unittest.TestCase):
    def test_warmup_reaches_base_lr_then_cosine_decays_to_eta_min(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)

        scheduler = EpochWarmupCosineScheduler(
            optimizer,
            total_epochs=5,
            warmup_epochs=2,
            eta_min=0.01,
            warmup_start_factor=0.1,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

        scheduler.step(1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

        scheduler.step(2)
        expected_epoch_3 = 0.01 + (0.1 - 0.01) * 0.5 * (
            1.0 + math.cos(math.pi / 3.0)
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected_epoch_3)

        scheduler.step(4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_state_dict_restores_epoch_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = EpochWarmupCosineScheduler(
            optimizer,
            total_epochs=5,
            warmup_epochs=2,
            eta_min=0.01,
            warmup_start_factor=0.1,
        )
        scheduler.step(3)
        saved_lr = optimizer.param_groups[0]["lr"]

        new_parameter = torch.nn.Parameter(torch.tensor(1.0))
        new_optimizer = torch.optim.AdamW([new_parameter], lr=0.1)
        new_scheduler = EpochWarmupCosineScheduler(
            new_optimizer,
            total_epochs=5,
            warmup_epochs=2,
            eta_min=0.01,
            warmup_start_factor=0.1,
        )

        new_scheduler.load_state_dict(scheduler.state_dict())

        self.assertEqual(new_scheduler.last_epoch, 3)
        self.assertAlmostEqual(new_optimizer.param_groups[0]["lr"], saved_lr)

    def test_factory_supports_existing_warm_restart_scheduler(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)

        scheduler = create_scheduler(
            optimizer,
            SCHEDULER_WARM_RESTART,
            t0=10,
            t_mult=2,
            eta_min=0.01,
            epochs=100,
            warmup_epochs=0,
            warmup_start_factor=0.1,
        )

        self.assertIsInstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        )

    def test_factory_supports_warmup_cosine_scheduler(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)

        scheduler = create_scheduler(
            optimizer,
            SCHEDULER_WARMUP_COSINE,
            t0=10,
            t_mult=2,
            eta_min=0.01,
            epochs=100,
            warmup_epochs=5,
            warmup_start_factor=0.1,
        )

        self.assertIsInstance(scheduler, EpochWarmupCosineScheduler)


if __name__ == "__main__":
    unittest.main()
