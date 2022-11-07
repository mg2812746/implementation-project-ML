# Libraries
import pandas as pd

# Gets data from file using pandas library
def store_data_from_file(FILE):
    try:
        data = pd.read_csv(f"adult_data/{FILE}", sep = ',',
                           header=None, names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','50k'])
        print(f"{FILE} read successfully")
        return data
    except ValueError:
        data = pd.read_csv(f"adult_data/{FILE}", sep = ',')
        print(f"{FILE} read successfully")
        return data
    except:
        print("something went wrong")

# Parse data by labels (columns)
def parse(LABEL, df):
    label = df[f'{LABEL}']
    return label
# Naive bayes algorithm
def naive_bayes(data_list, test_list):
    return None

# Program begins here

# Names of files we will read from
ADULT_DATA = 'adult.data'
TEST_DATA = 'adult.test'
# Labels associated with data
LABELS = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','50k']
# Get data from file
ad_df = store_data_from_file(ADULT_DATA)   
test_df = store_data_from_file(TEST_DATA)
# Parse data from pandas dataframes adult and test into appropriate variables
for labels in LABELS:
    locals()[labels] = parse(labels, ad_df)

# Parse data

# Naive bayes implementation
naive_bayes(ad_df, test_df)