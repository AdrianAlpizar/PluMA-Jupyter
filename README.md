# PluMA-Jupyter

## Clone the Repository

## Create Virtual Enviroment
### UNIX
`python3 -m venv env`<br />
`source env/bin/activate`<br />

### WINDOWS
`py -m venv env`<br />
`.\env\Scripts\activate`<br />

## Install Dependencies
`pip install -r requirements.txt`<br />

## Start working & fixing errors 
### After done, before pushing!
1. Make sure before comminting and pushing your changes you update the requirements.txt file<br />
`pipdeptree -f --warn silence | grep -E '^[a-zA-Z0-9\-]+' > requirements.txt`

