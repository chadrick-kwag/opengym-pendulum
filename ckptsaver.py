import os, torch

class PriorityQueue:

    def __init__(self, large_first = True):
        self.item_list = []
        self.large_first = large_first

    def __len__(self):

        return len(self.item_list)
    
    def add(self, item):

        self.item_list.append(item)

        self.item_list.sort(key=lambda x: x, reverse=True if self.large_first else False)

    def getbyindex(self, idx):

        return self.item_list[idx]
    
    def remove_index(self, idx):

        del self.item_list[idx]



class Saver:

    def __init__(self, savedir, keepsize=3, large_first = True):

        self.pq = PriorityQueue(large_first= large_first)
        self.savedir = savedir

        assert os.path.exists(savedir)

        self.keep_size= keepsize


    def add_new_metric(self, epi_index, metric, actor: torch.nn.Module):

        savepath = os.path.join(self.savedir, f'{epi_index}_metric={metric}.pt')

        self.pq.add((metric, savepath))

        save_current = True

        if len(self.pq) > self.keep_size:
            _, path = self.pq.getbyindex(self.keep_size) 
            self.pq.remove_index(self.keep_size)

            if path != savepath:
                os.remove(path)
            else:
                save_current = False
        else:
            save_current = True

        if save_current:
            torch.save(actor.state_dict(), savepath)


