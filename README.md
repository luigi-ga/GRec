# food_recommendation

Preliminary steps
1. Download neo4j 
2. Create a new DBMS (i.e. name it 'Food DBMS')
3. Add to the DBMS the APOC and GDS libraries
4. Start the DBMS

# Install all dependencies
```
pip install -r requirements
```


# Downloading Kaggle Dataset via Terminal

To download a dataset from Kaggle directly through the terminal, follow these steps. This method uses the Kaggle API, which requires authentication.

## Prerequisites

- Ensure you have Python and pip installed on your system.
- You will need a Kaggle account to proceed.

## Steps

### 1. Install Kaggle API (skip if u already installed all dipendencies)

First, install the Kaggle API using pip if it's not already installed:
```
pip install kaggle
```


### 2. API Credentials

Next, obtain your Kaggle API credentials:

- Log in to [Kaggle](https://www.kaggle.com) and navigate to your account settings (`https://www.kaggle.com/account`).
- Scroll to the "API" section and click "Create New API Token".
- This action downloads a `kaggle.json` file containing your API credentials.

### 3. Configure Kaggle API

Configure your system to use the API credentials by placing the `kaggle.json` file in a specific directory:
```
mkdir -p ~/.kaggle
cp path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Replace `path/to/kaggle.json` with the actual path to the file you downloaded.


