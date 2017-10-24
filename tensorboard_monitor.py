import tensorflow as tf
from monitor import Monitor

class TensorboardMonitor(Monitor):
    
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)     
        self.epoch = 0   

    def epoch_result(self, phase: str, epoch_loss: float, epoch_accuracy: float, roc_loss: float) -> None:
        self.write_scalar('epoch_loss', epoch_loss)
        self.write_scalar('epoch_accuracy', epoch_accuracy)
        self.write_scalar('epoch_roc_loss', roc_loss)

    def epoch_start(self, epoch: int, epoch_count: int) -> None:
        self.epoch = self.epoch + 1
        self.write_scalar('epoch', epoch)

    def training_completed(self, time_elapsed: float, best_loss: float, best_acc: float) -> None:
        self.write_scalar('best_loss', best_loss)
        self.write_scalar('best_acc', best_acc)
        self.write_scalar('time_elapsed', time_elapsed)

    def write_scalar(self, name: str, value: float):
        scalar = tf.Summary.Value(simple_value=value, tag=name)
        summary = tf.Summary(value=[scalar])
        self.writer.add_summary(summary, self.epoch)
