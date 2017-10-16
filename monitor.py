class Monitor(object):
    
    def epoch_result(self, phase: string, epoch_loss: float, epoch_accuracy: float) -> None:
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

    def epoch_start(self, epoch: int, epoch_count: int) -> None:
        print(f'Epoch {epoch}/{epoch_count}')