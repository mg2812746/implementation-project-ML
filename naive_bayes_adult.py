""" Naive Bayes Classifier Implementation of Adult Population Data

Summary:
    Trained by 32,561 instances. Goal is to classify whether the test 
    instance earns greater than or equal to 50k income, or less than 
    50k income.
    
"""

# Libraries
import pandas as pd

# Gets data from file using pandas library
def store_data_from_file(FILE, LABELS):
    try:
        data = pd.read_csv(f"adult_data/{FILE}", sep = ',',
                           header=None, names = LABELS)
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
def naive_bayes(ad_df, test_df, LABELS):
    def discretize(label, cutoff, name): #binary discretization procedure
        if name != None:
            # Discretize class label
            disc_label=label.where(label == name, 0)
            disc_label=disc_label.where(disc_label == 0, 1)
        else:
            # Discretize integer label
            disc_label=label.where(label < cutoff, 1)
            disc_label=disc_label.where(disc_label == 1, 0)
        return disc_label
    def prob(age,fnlwgt,edu,ms,oc,rel,rc,sx,ed_num,cap_gain,cap_loss,hrs_pr_wk,ntv_cntry,_50k,y_or_n): # Generate relevant probabilities
        INST=32561 # No of instances
        # P(X|>50k)
        a_cnt=0;f_cnt=0;e_cnt=0;m_cnt=0;o_cnt=0;r_cnt=0;rc_cnt=0;sx_cnt=0;edn_cnt=0;edn_cnt=0;cpgain_cnt=0
        cploss_cnt=0;hrs_cnt=0;nc_cnt=0
        # P(X|<=50k)
        a_cnt_n=0;f_cnt_n=0;e_cnt_n=0;m_cnt_n=0;o_cnt_n=0;r_cnt_n=0;rc_cnt_n=0;sx_cnt_n=0;edn_cnt_n=0;edn_cnt_n=0;cpgain_cnt_n=0
        cploss_cnt_n=0;hrs_cnt_n=0;nc_cnt_n=0
        if y_or_n:
            for a,f,e,m,o,r,rc,sx,edn,cpgain,cploss,hrs,nc,fif in zip(age,fnlwgt,edu,ms,oc,rel,rc,sx,ed_num,cap_gain,cap_loss,hrs_pr_wk,ntv_cntry,_50k):
                # Generate total count of each instance
                if a == 1 and fif == 1: a_cnt+=1 
                if f == 1 and fif == 1: f_cnt+=1
                if e == 1 and fif == 1: e_cnt+=1
                if m == 1 and fif == 1: m_cnt+=1
                if o == 1 and fif == 1: o_cnt+=1
                if r == 1 and fif == 1: r_cnt+=1
                if rc == 1 and fif == 1: rc_cnt+=1 
                if sx == 1 and fif == 1: sx_cnt+=1
                if edn == 1 and fif == 1: edn_cnt+=1
                if cpgain == 1 and fif == 1: cpgain_cnt+=1
                if cploss == 1 and fif == 1: cploss_cnt+=1
                if hrs == 1 and fif == 1: hrs_cnt+=1
                if nc == 1 and fif == 1: nc_cnt+=1
                if a == 1 and fif == 0: a_cnt_n+=1 
                if f == 1 and fif == 0: f_cnt_n+=1
                if e == 1 and fif == 0: e_cnt_n+=1
                if m == 1 and fif == 0: m_cnt_n+=1
                if o == 1 and fif == 0: o_cnt_n+=1
                if r == 1 and fif == 0: r_cnt_n+=1
                if rc == 1 and fif == 0: rc_cnt_n+=1 
                if sx == 1 and fif == 0: sx_cnt_n+=1
                if edn == 1 and fif == 0: edn_cnt_n+=1
                if cpgain == 1 and fif == 0: cpgain_cnt_n+=1
                if cploss == 1 and fif == 0: cploss_cnt_n+=1
                if hrs == 1 and fif == 0: hrs_cnt_n+=1
                if nc == 1 and fif == 0: nc_cnt_n+=1
        else:
            for a,f,e,m,o,r,rc,sx,edn,cpgain,cploss,hrs,nc,fif in zip(age,fnlwgt,edu,ms,oc,rel,rc,sx,ed_num,cap_gain,cap_loss,hrs_pr_wk,ntv_cntry,_50k):
                # Generate total count of each instance
                if a == 0 and fif == 1: a_cnt+=1 
                if f == 0 and fif == 1: f_cnt+=1
                if e == 0 and fif == 1: e_cnt+=1
                if m == 0 and fif == 1: m_cnt+=1
                if o == 0 and fif == 1: o_cnt+=1
                if r == 0 and fif == 1: r_cnt+=1
                if rc == 0 and fif == 1: rc_cnt+=1 
                if sx == 0 and fif == 1: sx_cnt+=1
                if edn == 0 and fif == 1: edn_cnt+=1
                if cpgain == 0 and fif == 1: cpgain_cnt+=1
                if cploss == 0 and fif == 1: cploss_cnt+=1
                if hrs == 0 and fif == 1: hrs_cnt+=1
                if nc == 0 and fif == 1: nc_cnt+=1
                if a == 0 and fif == 0: a_cnt_n+=1 
                if f == 0 and fif == 0: f_cnt_n+=1
                if e == 0 and fif == 0: e_cnt_n+=1
                if m == 0 and fif == 0: m_cnt_n+=1
                if o == 0 and fif == 0: o_cnt_n+=1
                if r == 0 and fif == 0: r_cnt_n+=1
                if rc == 0 and fif == 0: rc_cnt_n+=1 
                if sx == 0 and fif == 0: sx_cnt_n+=1
                if edn == 0 and fif == 0: edn_cnt_n+=1
                if cpgain == 0 and fif == 0: cpgain_cnt_n+=1
                if cploss == 0 and fif == 0: cploss_cnt_n+=1
                if hrs == 0 and fif == 0: hrs_cnt_n+=1
                if nc == 0 and fif == 0: nc_cnt_n+=1
        # Divide by total number of instances for label=1
        a_cnt/=INST; f_cnt/=INST; e_cnt/=INST; m_cnt/=INST; o_cnt/=INST; r_cnt/=INST
        rc_cnt/=INST; sx_cnt/=INST; edn_cnt/=INST; cpgain_cnt/=INST; cploss_cnt/=INST
        hrs_cnt/=INST; nc_cnt/=INST
        # Divide by total number of instances for label=0
        a_cnt_n/=INST; f_cnt_n/=INST; e_cnt_n/=INST; m_cnt_n/=INST; o_cnt_n/=INST; r_cnt_n/=INST
        rc_cnt_n/=INST; sx_cnt_n/=INST; edn_cnt_n/=INST; cpgain_cnt_n/=INST; cploss_cnt_n/=INST
        hrs_cnt_n/=INST; nc_cnt_n/=INST
        return a_cnt,f_cnt,e_cnt,m_cnt,o_cnt,r_cnt,rc_cnt,sx_cnt,\
                edn_cnt,edn_cnt,cpgain_cnt,cploss_cnt,hrs_cnt,nc_cnt,\
                a_cnt_n,f_cnt_n,e_cnt_n,m_cnt_n,o_cnt_n,\
                r_cnt_n,rc_cnt_n,sx_cnt_n,edn_cnt_n,\
                edn_cnt_n,cpgain_cnt_n,cploss_cnt_n,hrs_cnt_n,nc_cnt_n
    def gen_list_prob(LABELS,ad_df):
        # Parse data from pandas dataframes into parallel lists
        for label in LABELS: globals()[label] = parse(label, ad_df)
        # Discretize labels
        disc_age=discretize(age, 40, None) # 0 for <=40 1 for >40
        disc_fnlwgt=discretize(fnlwgt, 100000, None) # 0 for <=100000 1 for >100000
        disc_education=discretize(education,None,' Bachelors') # Bachelors (1) versus anything else(0)
        disc_marital_status=discretize(marital_status,None,' Divorced') # Divorced (1) versus anything else(0)
        disc_occupation=discretize(occupation,None,' Other-service') # Other service (1) versus anything else(0)
        disc_relationship=discretize(relationship,None,' Unmarried') # Never worked (1) versus anything else(0)
        disc_race=discretize(race,None,' White') # White (1) versus any other race(0)
        disc_sex=discretize(sex,None,' Male') # Male (1) versus Female(0)
        disc_education_num=discretize(education_num, 40, None) # 0 for <=40 1 for >40
        disc_capital_gain=discretize(capital_gain, 0, None) # any capital gain (1) or none (0)
        disc_capital_loss=discretize(capital_loss, 0, None) # any capital loss (1) or none (0)
        disc_hours_per_week=discretize(hours_per_week, 39, None)  # full time or part time
        disc_native_country=discretize(native_country,None,' United-States') # 0 if not from US, 1 if from US
        disc_50k=discretize(_50k,None,' >50K') # 1 for greater than 50k 0 for lower than 50k  
        # Generate all probabilities for each label (X)
        # P(X=1|>50k) and P(X=1|<=50k)
        a_cnt,f_cnt,e_cnt,m_cnt,o_cnt,r_cnt,rc_cnt,sx_cnt,edn_cnt,edn_cnt,cpgain_cnt,\
        cploss_cnt,hrs_cnt,nc_cnt,a_cnt_n,f_cnt_n,e_cnt_n,m_cnt_n,o_cnt_n,r_cnt_n,rc_cnt_n,sx_cnt_n,\
        edn_cnt_n,edn_cnt_n,cpgain_cnt_n,cploss_cnt_n,hrs_cnt_n,nc_cnt_n\
        =prob(disc_age,disc_fnlwgt,disc_education,disc_marital_status,disc_occupation,
            disc_relationship,disc_race,disc_sex,disc_education_num,
            disc_capital_gain, disc_capital_loss, disc_hours_per_week, 
            disc_native_country, disc_50k, True)
        # <=50k P(X=0|>50k) and P(X=0|<=50k)
        a_cnt_0,f_cnt_0,e_cnt_0,m_cnt_0,o_cnt_0,r_cnt_0,rc_cnt_0,sx_cnt_0,edn_cnt_0,edn_cnt_0,cpgain_cnt_0,\
        cploss_cnt_0,hrs_cnt_0,nc_cnt_0,a_cnt_n_0,f_cnt_n_0,e_cnt_n_0,m_cnt_n_0,o_cnt_n_0,\
        r_cnt_n_0,rc_cnt_n_0,sx_cnt_n_0,edn_cnt_n_0,edn_cnt_n_0,cpgain_cnt_n_0,\
        cploss_cnt_n_0,hrs_cnt_n_0,nc_cnt_n_0\
        =prob(disc_age,disc_fnlwgt,disc_education,disc_marital_status,
            disc_occupation,disc_relationship,disc_race,disc_sex,disc_education_num,
            disc_capital_gain, disc_capital_loss, disc_hours_per_week, disc_native_country, disc_50k, False)
        # Store probabilities in appropriate lists
        LIST_P_GT50K_1=[a_cnt,f_cnt,e_cnt,m_cnt,o_cnt,r_cnt,rc_cnt,sx_cnt,edn_cnt,edn_cnt,cpgain_cnt,
                        cploss_cnt,hrs_cnt,nc_cnt]
        LIST_P_LT50K_1=[a_cnt_n,f_cnt_n,e_cnt_n,m_cnt_n,o_cnt_n,r_cnt_n,rc_cnt_n,sx_cnt_n,
                        edn_cnt_n,edn_cnt_n,cpgain_cnt_n,cploss_cnt_n,hrs_cnt_n,nc_cnt_n]
        LIST_P_GT50K_0=[a_cnt_0,f_cnt_0,e_cnt_0,m_cnt_0,o_cnt_0,r_cnt_0,rc_cnt_0,sx_cnt_0,edn_cnt_0,
                        edn_cnt_0,cpgain_cnt_0,cploss_cnt_0,hrs_cnt_0,nc_cnt_0]
        LIST_P_LT50K_0=[a_cnt_n_0,f_cnt_n_0,e_cnt_n_0,m_cnt_n_0,o_cnt_n_0,
                        r_cnt_n_0,rc_cnt_n_0,sx_cnt_n_0,edn_cnt_n_0,edn_cnt_n_0,cpgain_cnt_n_0,
                        cploss_cnt_n_0,hrs_cnt_n_0,nc_cnt_n_0]
        return LIST_P_GT50K_1,LIST_P_LT50K_1,LIST_P_GT50K_0,LIST_P_LT50K_0
    def test(LABELS,test_df): #Classifies >50k or <=50k over all test instances
        for label in LABELS: globals()[f"test_{label}"] = parse(label, test_df)
        # Discretize test labels
        test_disc_age=discretize(test_age, 40, None) # 0 for <=40 1 for >40
        test_disc_fnlwgt=discretize(test_fnlwgt, 100000, None) # 0 for <=100000 1 for >100000
        test_disc_education=discretize(test_education,None,' Bachelors') # Bachelors (1) versus anything else(0)
        test_disc_marital_status=discretize(test_marital_status,None,' Divorced') # Divorced (1) versus anything else(0)
        test_disc_occupation=discretize(test_occupation,None,' Other-service') # Other service (1) versus anything else(0)
        test_disc_relationship=discretize(test_relationship,None,' Unmarried') # Never worked (1) versus anything else(0)
        test_disc_race=discretize(test_race,None,' White') # White (1) versus any other race(0)
        test_disc_sex=discretize(test_sex,None,' Male') # Male (1) versus Female(0)
        test_disc_education_num=discretize(test_education_num, 40, None) # 0 for <=40 1 for >40
        test_disc_capital_gain=discretize(test_capital_gain, 0, None) # any capital gain (1) or none (0)
        test_disc_capital_loss=discretize(test_capital_loss, 0, None) # any capital loss (1) or none (0)
        test_disc_hours_per_week=discretize(test_hours_per_week, 39, None)  # full time or part time
        test_disc_native_country=discretize(test_native_country,None,' United-States') # 0 if not from US, 1 if from US
        test_disc_50k=discretize(test__50k,None,' >50K') # 1 for greater than 50k 0 for lower than 50k  
        return None
    def test_label(test_label,LIST_P): # Search function that returns needed probabilities from list
        # P(>50k|test_label)
        return None  
    def error():
        return None
    LIST_P_GT50K_1,LIST_P_LT50K_1,LIST_P_GT50K_0,LIST_P_LT50K_0=gen_list_prob(LABELS,ad_df)
# Program begins here

# Names of files we will read from
ADULT_DATA = 'adult.data'
TEST_DATA = 'adult.test'
# Labels associated with data
LABELS = ['age','workclass','fnlwgt','education','education_num',
          'marital_status','occupation','relationship','race',
          'sex','capital_gain','capital_loss','hours_per_week',
          'native_country','_50k']
# Get data from file
ad_df = store_data_from_file(ADULT_DATA, LABELS)   
test_df = store_data_from_file(TEST_DATA, LABELS)
# Naive bayes implementation
naive_bayes(ad_df, test_df, LABELS)