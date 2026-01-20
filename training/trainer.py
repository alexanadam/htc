import os
import torch


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        logger,
        train_loader,
        val_loader=None,
        evaluator=None,
        scheduler=None,
        grad_clip=None,
        checkpoint_dir=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._resume("last.pt")

        self.start_epoch = 1

        self.logger.info("Trainer initialized")

    def _resume(self, name):
        ckpt_path = os.path.join(self.checkpoint_dir, name)

        if not os.path.exists(ckpt_path):
            self.logger.info("No checkpoint found. Training from scratch.")
            return

        self.logger.info(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1

    def _save_checkpoint(self, epoch, name):
        self.logger.info(f"Epoch {epoch}: Saving {name} model...")
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }

        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()

        path = os.path.join(self.checkpoint_dir, name)
        torch.save(ckpt, path)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        aux_loss = 0.0

        for batch_id, batch in enumerate(self.train_loader, start=1):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)
            try:
                loss, extras = losses
            except TypeError:
                loss = losses
                extras = None
            loss.backward()

            # if self.grad_clip is not None:
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            running_loss += loss.item()
            if extras:
                aux_loss += extras.item()

            if batch_id % 50 == 1:
                self.logger.info(
                    f"Epoch {epoch}, "
                    f"Batch {batch_id}/{len(self.train_loader)}, "
                    f"Loss: {running_loss / batch_id:.4f}"
                )

        avg_loss = running_loss / len(self.train_loader)
        if aux_loss:
            self.logger.info(
                f"Epoch {epoch}: Aux Loss: {aux_loss/len(self.train_loader):.4f}"
            )
        return avg_loss

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        aux_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets.to(self.device)

                outputs = self.model(inputs)
                losses = self.criterion(outputs, targets)
                try:
                    loss, extras = losses
                except TypeError:
                    loss = losses
                    extras = None

                running_loss += loss.item()
            if extras:
                aux_loss += extras.item()

        avg_loss = running_loss / len(self.val_loader)
        if aux_loss:
            self.logger.info(
                f"Val aux loss: {aux_loss/len(self.val_loader):.4f}"
            )
        return avg_loss

    def train(self, epochs):
        # best_val_loss = float("inf")
        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self._train_epoch(epoch)
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")

            if self.val_loader is not None:
                val_loss = self._validate_epoch()
                self.logger.info(f"Epoch {epoch}: Val Loss: {val_loss:.4f}")

                if self.evaluator is not None:
                    metrics = self.evaluator(
                        model=self.model,
                        dataloader=self.val_loader,
                        device=self.device,
                    )

                    self.logger.info(
                        f"Epoch {epoch}: "
                        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    )

                if self.scheduler:
                    self.scheduler.step()

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(epoch, "last.pt")
                    # if val_loss < best_val_loss:
                    #     best_val_loss = val_loss
                    #     self._save_checkpoint(epoch, "best.pt")
