# Single-Cell-Autoencoders
The run.py file is our main file, and running the test target on it will train the models as well as output the loss of the test data. <br />
To obtain the data, we used https://openproblems.bio/neurips_docs/data/dataset/ . But in our github we will work with a smaller dataset in order to test out our targets.<br />
At the moment, the only command line argument for targets is "test." Running "python run.py test" will run the project as is and output what we have. Due to the <br />
current constraints on the size of files on GitHub, we only used a small subset of the data.
We have also uploaded a notebook which creates our visuals and also shows how to create our model and train it. The visualizations included in this notebook <br />
use a larger subset of data, as the smaller subset would produce worse plots.
