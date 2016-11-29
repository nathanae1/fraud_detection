# Fraud Detection Scoping Document

### Functional Scope:
The fraud detection application includes a number of key functionalities:
* Reads in data from server, performs prediction, and stores record in a database.
* Queries database, and populated dashboard page with informative tables and graphics
* To perform model training and prediction, a preprocessing script and a build model script will also be included in the scope.

### Data Scope:
The training data will be persisted in a json file on a server. The new data received from the network will be stored in mongodb after preprocessing and prediction has been performed.

### Technical Scope:
The majority of the project will be implemented in python. The preprocessing and build model scripts will utilize pandas data management and sklearn for model training. The build model script will also pass the trained model to the application using cPickle.

The application will be implemented using the python flask framework.

### Responsibilities Scope:
The responsibilities will be divided amongst the team members. The model training and development will be handled by Lee and Nelly. Nathan will handle EDA and feature generation, as well as writing the preprocessing function. Nathanael will be responsible for setting up the database, connecting the application to the broadcast server, and setting up an empty dashboard. Nelly will complete the features of the dashboard, and the rest of the team will help once their respective tasks are complete.
