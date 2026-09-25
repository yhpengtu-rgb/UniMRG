from mmengine.hooks import CheckpointHook


class StableCheckpointHook(CheckpointHook):
    """Place stable token provenance in DeepSpeed client_state metadata."""

    @staticmethod
    def stable_meta(runner, meta):
        metadata = getattr(runner, 'stable_tokenizer_metadata', None)
        if metadata is None:
            raise RuntimeError(
                'stable tokenizer metadata is unavailable at checkpoint time'
            )
        augmented = dict(meta)
        augmented['harmon_tokenizer'] = dict(metadata)
        return augmented

    def _save_checkpoint_with_step(self, runner, step, meta):
        return super()._save_checkpoint_with_step(
            runner,
            step,
            meta=self.stable_meta(runner, meta),
        )
