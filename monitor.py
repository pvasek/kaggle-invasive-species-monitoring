class Monitor(object):
    def epoch_result(self, phase: str, epoch_loss: float, epoch_accuracy: float) -> None:
        pass

    def epoch_start(self, epoch: int, epoch_count: int) -> None:
        pass

    def training_completed(self, time_elapsed: float, best_loss: float, best_acc: float) -> None:
        pass
    

class PrintMonitor(Monitor):
    
    def epoch_result(self, phase: str, epoch_loss: float, epoch_accuracy: float) -> None:
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

    def epoch_start(self, epoch: int, epoch_count: int) -> None:
        print(f'Epoch {epoch}/{epoch_count}')

    def training_completed(self, time_elapsed: float, best_loss: float, best_acc: float) -> None:
        print(f'Training completed in time {time_elapsed:.0f} seconds. Best loss: {best_loss}, Best acc: {best_acc}')