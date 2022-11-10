""" Naive Bayes Classifier Implementation of Adult Population Data

Summary:
    Trained by 32,561 instances. Goal is to classify whether the test 
    instance earns greater than or equal to 50k income, or less than 
    50k income.
    
"""

# Libraries
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
# Gets data from file using pandas library
def store_data_from_file(FILE, FEATURES):
    try:
        data = pd.read_csv(f"adult_data/{FILE}", sep = ',',
                           header=None, names = FEATURES)
        print(f"{FILE} read successfully")
        return data
    except ValueError:
        data = pd.read_csv(f"adult_data/{FILE}", sep = ',')
        print(f"{FILE} read successfully")
        return data
    except:
        print("something went wrong")

# Parse data by features (columns)
def parse(FEATURE, df):
    feature = df[f'{FEATURE}']
    return feature

# Naive bayes algorithm
def naive_bayes(ad_df, test_df, FEATURES):
    def discretize(label, cutoff, name): #binary discretization procedure
        if name != None:
            # Discretize class feature
            disc_label=label.where(label == name, 0)
            disc_label=disc_label.where(disc_label == 0, 1)
        else:
            # Discretize integer feature
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
            x=1
        else:
            x=0
        for a,f,e,m,o,r,rc,sx,edn,cpgain,cploss,hrs,nc,fif in zip(age,fnlwgt,edu,ms,oc,rel,rc,sx,ed_num,cap_gain,cap_loss,hrs_pr_wk,ntv_cntry,_50k):
            # Generate total count of each instance
            if a == x and fif == 1: a_cnt+=1 
            if f == x and fif == 1: f_cnt+=1
            if e == x and fif == 1: e_cnt+=1
            if m == x and fif == 1: m_cnt+=1
            if o == x and fif == 1: o_cnt+=1
            if r == x and fif == 1: r_cnt+=1
            if rc == x and fif == 1: rc_cnt+=1 
            if sx == x and fif == 1: sx_cnt+=1
            if edn == x and fif == 1: edn_cnt+=1
            if cpgain == x and fif == 1: cpgain_cnt+=1
            if cploss == x and fif == 1: cploss_cnt+=1
            if hrs == x and fif == 1: hrs_cnt+=1
            if nc == x and fif == 1: nc_cnt+=1
            if a == x and fif == 0: a_cnt_n+=1 
            if f == x and fif == 0: f_cnt_n+=1
            if e == x and fif == 0: e_cnt_n+=1
            if m == x and fif == 0: m_cnt_n+=1
            if o == x and fif == 0: o_cnt_n+=1
            if r == x and fif == 0: r_cnt_n+=1
            if rc == x and fif == 0: rc_cnt_n+=1 
            if sx == x and fif == 0: sx_cnt_n+=1
            if edn == x and fif == 0: edn_cnt_n+=1
            if cpgain == x and fif == 0: cpgain_cnt_n+=1
            if cploss == x and fif == 0: cploss_cnt_n+=1
            if hrs == x and fif == 0: hrs_cnt_n+=1
            if nc == x and fif == 0: nc_cnt_n+=1
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
    def gen_list_prob(FEATURES,ad_df): #Creates a list of probabilities
        # Parse data from pandas dataframes into dynamically created parallel lists
        for label in FEATURES: globals()[label] = parse(label, ad_df)
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
    def prob_from_list(LIST_P,index): # Search function that returns needed probabilities from list
        """   INDICES ARE AS FOLLOWS:
                0=age, 1=fnlwgt, 2=education, 3=education_num, 4=marital-status, 5=occupation
                6=relationship, 7=race , 8=sex, 9=capital-gain, 10=capital-loss, 11=hours-per-week, 12=native-country
        """
        return LIST_P[index]  
    def test(FEATURES,test_df,P_GT50K_1,P_LT50K_1,P_GT50K_0,P_LT50K_0): #Returns a list of all classified test instances for >50k or <=50k
        classified=[]
        for feature in FEATURES: globals()[f"test_{feature}"] = parse(feature, test_df)
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
        test_disc_50k=discretize(test__50k,None,' >50K') # THESE WILL BE USED TO COMPARE COMPUTED CLASSIFICATION VS ACTUAL 
        # Traverse through all test instances and classify each one, which is then stored into a list and then return that list
        for a,f,e,m,o,r,rc,sx,edn,cpgain,cploss,hrs,nc in zip(test_disc_age,test_disc_fnlwgt,
                                                                  test_disc_education,test_disc_marital_status,test_disc_occupation,
                                                                  test_disc_relationship,test_disc_race,test_disc_sex,test_disc_education_num,
                                                                  test_disc_capital_gain,test_disc_capital_loss,
                                                                  test_disc_hours_per_week,test_disc_native_country):
            prob_yes=[i for i in range(13)]
            prob_no=[i for i in range(13)]
            # assign probability of >50k income (yes) and <=50k income (no) given age>40 to list
            if a == 1 :  prob_yes[0]=prob_from_list(P_GT50K_1,0); prob_no[0]=prob_from_list(P_LT50K_1,0)
            else: prob_yes[0]=prob_from_list(P_GT50K_0,0); prob_no[0]=prob_from_list(P_LT50K_0,0)      
            if f == 1 : prob_yes[1]=prob_from_list(P_GT50K_1,1); prob_no[1]=prob_from_list(P_LT50K_1,1)
            else: prob_yes[1]=prob_from_list(P_GT50K_0,1); prob_no[1]=prob_from_list(P_LT50K_0,1)
            if e == 1 : prob_yes[2]=prob_from_list(P_GT50K_1,2); prob_no[2]=prob_from_list(P_LT50K_1,2)
            else: prob_yes[2]=prob_from_list(P_GT50K_0,2); prob_no[2]=prob_from_list(P_LT50K_0,2)
            if m == 1 : prob_yes[3]=prob_from_list(P_GT50K_1,3); prob_no[3]=prob_from_list(P_LT50K_1,3)
            else: prob_yes[3]=prob_from_list(P_GT50K_0,3); prob_no[3]=prob_from_list(P_LT50K_0,3)
            if o == 1 : prob_yes[4]=prob_from_list(P_GT50K_1,4); prob_no[4]=prob_from_list(P_LT50K_1,4)
            else: prob_yes[4]=prob_from_list(P_GT50K_0,4); prob_no[4]=prob_from_list(P_LT50K_0,4)
            if r == 1 : prob_yes[5]=prob_from_list(P_GT50K_1,5); prob_no[5]=prob_from_list(P_LT50K_1,5)
            else: prob_yes[5]=prob_from_list(P_GT50K_0,5); prob_no[5]=prob_from_list(P_LT50K_0,5)
            if rc == 1 :prob_yes[6]=prob_from_list(P_GT50K_1,6); prob_no[6]=prob_from_list(P_LT50K_1,6)
            else: prob_yes[6]=prob_from_list(P_GT50K_0,6); prob_no[6]=prob_from_list(P_LT50K_0,6)
            if sx == 1 : prob_yes[7]=prob_from_list(P_GT50K_1,7); prob_no[7]=prob_from_list(P_LT50K_1,7)
            else: prob_yes[7]=prob_from_list(P_GT50K_0,7); prob_no[7]=prob_from_list(P_LT50K_0,7)
            if edn == 1 : prob_yes[8]=prob_from_list(P_GT50K_1,8); prob_no[8]=prob_from_list(P_LT50K_1,8)
            else: prob_yes[8]=prob_from_list(P_GT50K_0,8); prob_no[8]=prob_from_list(P_LT50K_0,8)
            if cpgain == 1 : prob_yes[9]=prob_from_list(P_GT50K_1,9); prob_no[9]=prob_from_list(P_LT50K_1,9)
            else: prob_yes[9]=prob_from_list(P_GT50K_0,9); prob_no[9]=prob_from_list(P_LT50K_0,9)
            if cploss == 1 : prob_yes[10]=prob_from_list(P_GT50K_1,10); prob_no[10]=prob_from_list(P_LT50K_1,10)
            else: prob_yes[10]=prob_from_list(P_GT50K_0,10); prob_no[10]=prob_from_list(P_LT50K_0,10)
            if hrs == 1 : prob_yes[11]=prob_from_list(P_GT50K_1,11); prob_no[11]=prob_from_list(P_LT50K_1,11)
            else: prob_yes[11]=prob_from_list(P_GT50K_0,11); prob_no[11]=prob_from_list(P_LT50K_0,11)
            if nc == 1 : prob_yes[12]=prob_from_list(P_GT50K_1,12); prob_no[12]=prob_from_list(P_LT50K_1,12)
            else: prob_yes[12]=prob_from_list(P_GT50K_0,12); prob_no[12]=prob_from_list(P_LT50K_0,12)
            # After we get our probabilities, we compare the probability of yes vs no overall 
            # Then classify based on which probability is greater
            prob_yes_fin=0
            prob_no_fin=0
            avoid_yes=0
            avoid_no=0
            for i in range(0,len(prob_yes)):
                if(prob_yes[i]!=0):
                    prob_yes_fin=prob_yes[i]
                    avoid_yes=i
                    break
            for i in range(0,len(prob_no)):
                if(prob_yes[i]!=0):
                    prob_no_fin=prob_no[i]
                    avoid_no=i
                    break
            for i in range(1,len(prob_yes)):
                if(prob_yes[i]!=0 and avoid_yes != i):
                    temp=prob_yes[i]
                    if(prob_yes_fin*temp!=0):
                        prob_yes_fin=prob_yes_fin*temp
                    else: 
                        break
                if(prob_no[i]!=0 and avoid_no != i):
                    temp=prob_no[i]
                    if(prob_no_fin*temp!=0):
                        prob_no_fin=prob_no_fin*temp
                    else: 
                        break
                if(i==12 and prob_yes_fin > prob_no_fin):
                    classified.append(1)
                    
                if(i==12 and prob_no_fin >= prob_yes_fin):
                    classified.append(0)
        return classified, test_disc_50k
    def error(test_disc_50k, classified_50k): # Returns percent error
        INST=16281
        num_errors=0
        for i,j in zip(test_disc_50k,classified_50k):
            if i!=j: num_errors+=1
        return (num_errors/INST)*100 # Return percent error
    # Get list of all probabilities
    P_GT50K_1,P_LT50K_1,P_GT50K_0,P_LT50K_0=gen_list_prob(FEATURES,ad_df)
    # Get list of classified values from test and actual results
    classified, test_disc_50k=test(FEATURES,test_df,P_GT50K_1,P_LT50K_1,P_GT50K_0,P_LT50K_0)
    # Convert pandas dataframe to list
    t_list=test_disc_50k.tolist()
    # Compute percent error
    errors=error(test_disc_50k, classified)
    # Compute accuracy
    Accuracy = metrics.accuracy_score(t_list, classified)
    # Compute confusion matrix
    confusion_matrix = metrics.confusion_matrix(t_list, classified)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    print(f"{Accuracy:.2f}")
    
# Program begins here
def main():
    # Names of files we will read from
    ADULT_DATA = 'adult.data'
    TEST_DATA = 'adult.test'
    # Features associated with data
    FEATURES = ['age','workclass','fnlwgt','education','education_num',
            'marital_status','occupation','relationship','race',
            'sex','capital_gain','capital_loss','hours_per_week',
            'native_country','_50k']
    # Get data from file
    ad_df = store_data_from_file(ADULT_DATA, FEATURES)   
    test_df = store_data_from_file(TEST_DATA, FEATURES)
    # Naive bayes implementation
    naive_bayes(ad_df, test_df, FEATURES)

if __name__ == "__main__":
    main()
