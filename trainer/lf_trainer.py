from trainer.trainer import Trainer
import torch
import numpy as np

class LFTrainer(Trainer):
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(LFTrainer, self).__init__(model, loss, metrics, resume, config, data_loader, valid_data_loader, train_logger)
        self.end_loss_weight = config['end_loss_weight'] if 'end_loss_weight' in config else 1.0

    def _train_iteration(self, iteration):
        """
        Training logic for an iteration

        :param iteration: Current training iteration.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        #tic=timeit.default_timer()
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            data, positions_xyxy, positions_xyrs, step_count, forwards, detected_end_points = self._to_tensor(*self.data_loader_iter.next())
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            data, positions_xyxy, positions_xyrs, step_count, forwards, detected_end_points = self._to_tensor(*self.data_loader_iter.next())
        #toc=timeit.default_timer()
        #print('data: '+str(toc-tic))

        #tic=timeit.default_timer()

        self.optimizer.zero_grad()

        #if self.
        #if self.detectorModel is not None:
        #    linePreds, pointPreds, pixelPreds = self.detectorModel(data
        #step_count=len(positions_xyrs)
        #print(step_count)
        rand=True
        output,outputrs,output_end = self.model(
                data,
                positions_xyrs[0], 
                forwards, 
                steps=step_count, 
                all_positions=positions_xyrs, 
                all_xy_positions=positions_xyxy, 
                reset_interval=4, 
                randomize=rand, 
                skip_grid=True, 
                detected_end_points=detected_end_points)
        pos_loss = self.loss['pos'](output, positions_xyxy)
        if len(output_end)>0:
            end_loss = self.end_loss_weight*self.loss['end'](output_end,output,positions_xyrs[-1][:,0:2])
        else:
            end_loss=0
        #loss = self.loss(outputrs, positions_xyrs)
        loss = pos_loss + end_loss
        loss.backward()
        self.optimizer.step()

        #toc=timeit.default_timer()
        #print('for/bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics = self._eval_metrics(output, positions_xyxy)
        #toc=timeit.default_timer()
        #print('metric: '+str(toc-tic))

        #tic=timeit.default_timer()
        loss = loss.item()
        #toc=timeit.default_timer()
        #print('item: '+str(toc-tic))


        log = {
            'loss': loss,
            'pos_loss': pos_loss.item(),
            'end_loss': end_loss.item(),
            'metrics': metrics
        }


        return log


    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_pos_loss = 0
        total_end_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, inst in enumerate(self.valid_data_loader):
                data, positions_xyxy, positions_xyrs, step_count, forwards, detected_end_points = self._to_tensor(*inst)

                output,outputrs,output_end = self.model(
                        data,
                        positions_xyrs[0],
                        forwards,
                        steps=step_count,
                        skip_grid=True,
                        detected_end_points=detected_end_points)
                pos_loss = self.loss['pos'](output, positions_xyxy)
                if len(output_end)>0:
                    end_loss = self.end_loss_weight * self.loss['end'](output_end,output,positions_xyrs[-1][:,0:2])
                else:
                    end_loss=0
                loss = pos_loss + end_loss
                #loss = self.loss(outputrs, positions_xyrs)

                total_val_loss += loss.item()
                total_pos_loss += pos_loss.item()
                total_end_loss += end_loss.item()
                total_val_metrics += self._eval_metrics(output, positions_xyxy)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_pos_loss': total_pos_loss / len(self.valid_data_loader),
            'val_end_loss': total_end_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
