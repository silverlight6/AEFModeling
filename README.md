# AEFModeling

Instructions on how to train this model are included in the monthly reports.
This model is trained on student data to identify either [focused, nan, distracted] or [stable, transition]
This repository contains code for both a VIT version of the model as well as one for ConvTran.
Vast majority of ConvTran code taken directly from the ConvTran Github https://github.com/Navidfoumani/ConvTran

Instructions on how to run this model with on unlabeled data.

1. Download your data onto your local machine and clone this repository.
2. In the dataloader file -> in the change_set_minus_labels method -> change the path to be the full path where the data is (currently line 201)
3. Change the save path to a folder of your choice. (currently line 239)
4. Run the change_set_minus_labels. If you have a very large dataset, this may take some time. 
	On individual time stamps like it will be in the real world inference, it should take no time at all.
5. Change the path in the convTran file in the convTranWOLabels method to be your current path.
6. Change the number of workers to be 80% of your current cpu count.
7. Download the model from the Drive and change the save_path to point to that model.
8. Change the save path for the final outputs to where you would like the outputs to be
9. Change main.py to run the desired method. 
