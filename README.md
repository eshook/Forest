# ForEST
ForEST is a domain-specific language FOR Expressing Spatial-Temporal (FOREST) computation. Essentially, for GIScientists ForEST makes it easy to parallelize spatial-temporal computing using Python as the host language. For computational scientists, it makes it easy to test new ways to parallelize spatial and spatial-temporal methods using spatial domain decomposition or functional decomposition. 

## ForEST refactoring for growth
Sorry if things are broken right now. We are refactoring the code in ForEST. To make code commitments from external projects easier (most behind the scenes right now) we are changing some of the file structure and import process in ForEST. Right now the public repository is a bit behind and slightly broken. We are retooling and refactoring the 'core' of ForEST versus the 'add-ons' and extra 'capabilities' for different projects. This will streamline the development process without requiring a growing list of package dependencies. Hopefully we will have this fixed in a week or three, but this will make it much easier for ForEST to grow and interface with other packages such as [PySAL](https://pysal.readthedocs.io/en/latest/) so we can play nice with the rich (open source and commercial) geospatial software ecosystem.

## ForEST in use
Tyler Buresh, a Computer Science undergraduate student who graduated in Spring 2019, helped develop several key advances in ForEST. We accepted his pull request that has all of his contributions including a CUDA Engine in ForEST to run simulations models on a GPU. He also was the first to test the iterative function in ForEST. He produced two models: Game-of-Life and a species distribution model for the invasive Brown Marmorated Stink Bug.

We are now looking to repeat this for the Minnesota field mapping project. We are in the process of adapting Python codes that have processed statewide satellite imagery to produce a map of all farm fields in the state of Minnesota. The code will be shifted to ForEST syntax to test the ability of ForEST to capture these spatial analytics in an entire workflow. Scalability and performance will also be a key indicator.

Several other projects are in the pipeline, but are not public yet. So stay tuned...

## ForEST on the Road

Dr. Eric Shook, lead ForEST designer and developer, just returned from the road. We were finally able to publicly showcase the underlying programming model for ForEST called the Space-Time Key-Collection Programming model. Please see our manuscript entitled ["Space-Time is the Key: For Expressing Spatial-Temporal Computation"](https://auckland.figshare.com/articles/Space-Time_is_the_Key_For_Expressing_Spatial-Temporal_Computing/9870416) published in the *Proceedings for the 15th International Conference on Geocomputation* (see proceedings in the following link: [https://auckland.figshare.com/GeoComputation2019](https://auckland.figshare.com/GeoComputation2019)). We were also honored to be invited to be a plenary speaker for the [New Zealand Geospatial Research Conference 2019](https://geospatial.ac.nz/nzgrc-2019/). Our talk featured ForEST and the US National Science Foundation-supported Hour of CI project and how they are ["Lowering the Barriers to Scalable Geospatial Computation"](https://geospatial.ac.nz/nzgrc-2019-abstracts/#EricShook). This work would not have been possible without the help of many collaborators and students. 

## Whiteboard to publication
If you have been watching the ForEST page you will see that we had a break in development. First, we paused development to create a theoretical model that was just published called the Space-Time Key-Collection model. We then redesigned ForEST around the model. We have also been working in several private repo's to align ForEST with some of our collaborators codes. Unfortunately, most of this development work is out in the open yet so keep checking back. Please note that we released the GPU/CUDA Engine and demonstration models as part of the invasive species modeling dispersal project.

<pre>
  __________________    
 |                  |   Our whiteboarding efforts have finally been published!
 | def forest():    |   Check out our Geocomputation 2019 paper for highlights.
 |     awesome=True |   Follow on articles are underway with even more details.
 |__________________|   
  ==================
 </pre>

## Funded ForEST projects
Our work using using ForEST to develop a spatial dispersal model of an invasive species in the state of Minnesota, the Brown Marmorated Stink Bug, has been uploaded to the public repo. Much of this work was completed by Tyler Buresh, a talented undergraduate student from the University of Minnesota, under the supervision of Dr. Bryan Runck. We now have a functioning GPU engine using CUDA that supports the foundational Game-of-Life model as well as the spatial dispersal model for the Brown Marmorated Stink Bug.

Our project  using ForEST to analyze satellite imagery to identify farm fields was funded by the Center for Urban and Regional Affairs at the University of Minnesota. Our first ForEST prototype implementation will be uploaded to the public repo in the next few months. We shared a snapshot of our work at the [GeoAI Trillion Pixel Challenge Workshop](https://geoai.ornl.gov/trillion-pixel/) where Dr. Shook was session moderator for the Hardware Design and High-Performance Computing for GeoAI session.
