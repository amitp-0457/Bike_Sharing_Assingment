# Project Name - Bike Sharing System
This assignment is a programming assignment wherein we have to build a multiple linear regression model for the prediction of demand for shared bikes.

 
## Table of Contents
* [General Info]
* [Technologies Used] panda,matplotlib,seaborn
* [Conclusions]
* [Acknowledgements]

## General Information

Problem Statement
   A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or
   free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters
   the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.

   A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The
   company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business
   plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state.

   In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine 
   situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the 
   situation gets better all around and stand out from other service providers and make huge profits.

   They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically,
   they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:
   
       -Which variables are significant in predicting the demand for shared bikes.
       -How well those variables describe the bike demands
       -Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike
        demands across the American market based on some factors. 

Business Goal:
       We are required to model the demand for shared bikes with the available independent variables. It will be used by the management to      
       understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the
       demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand
       dynamics of a new market. 


- What is the dataset that is being used?

=========================================
Dataset characteristics
=========================================	
day.csv have the following fields:
	
	- instant: record index
	- dteday : date
	- season : season (1:spring, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2018, 1:2019)
	- mnth : month ( 1 to 12)
	- holiday : weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : temperature in Celsius
	- atemp: feeling temperature in Celsius
	- hum: humidity
	- windspeed: wind speed
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered
	

## Conclusions
The solution is divided into the following sections:

    Data understanding and exploration
    Data Visualisation
    Data preparation
    Model building and evaluation

- Before building model following sanity checks were done
   
    -Findings on raw data check:
           Dataset has 730 rows and 16 columns. Except one column(dteday), all other are either float or integer type. dte column is date
           type.There are some columns which are categorical in nature.We will analyse and finalize whether to convert them to categorical
           or treat as integer.
    
   - Data quality check on missing values 
           There are no missing/ Null values either in columns or rows.
           There were zero duplicate values in the dataset.
           There seems to be no Junk/Unknown values in the entire dataset.
    
- Understanding the Data Dictionary and parts of Data Preparation

      The data dictionary contains the meaning of various attributes; some of which are explored and manipulated here:
      The following variables can be removed from further analysis:

         instant : Its only an index value

         dteday:This has the date, Since we already have seperate columns for 'year' & 'month',hence it is not required

         casual & registered : Both these columns contains the count of bike booked by different categories of customers. Since our
         objective is to find the total count of bikes and not by specific category, we will ignore these two columns.

- Creating Dummy Variables
         Create DUMMY variables for 4 categorical variables 'mnth', 'weekday', 'season' & 'weathersit'.

- Understanding the data - Following need to be done :-
        Understanding the distribution of various numeric variables 
        If there is some obvious multicollinearity going on, this is the first place to catch it
        Here's where we will also identify if some predictors directly have a strong association with the outcome variable

- Observations related to Categorical Data
    There were 6 categorical variables in the dataset.

       Need to study their effect on the dependent variable (‘cnt’) .

       season: Around 32% of the bike booking were happening in fall with a median of over 5000 booking (for the period of 2 years). This
               was followed by summer & winter with 27% & 25% of total booking. This indicates season can be a good predictor for the 
               dependent variable.

       mnth: Almost 10% of the bike booking were happening in the months "may,june,july,august & sept" (5,6,7,8 & 9) with a median of over
             4000 booking per month. This indicates mnth can be a good predictor for the dependent variable.
      
       weathersit: Almost 67% of the bike booking were happening during ‘weathersit-1" ie "A" with a median of close to 5000 booking (for
                   the period of 2 years). This was followed by "weathersit-2" ie "B" with 30% of total booking. This indicates,
                   weathersit does show some trend towards the bike bookings can be a good predictor for the dependent variable.

       holiday: Almost 98% of the bike booking were happening when it is not a holiday which means this data is clearly biased. This
                indicates, holiday cannot be a good predictor for the dependent variable.

       weekday: weekday variable shows very close trend on all days of the week having their medians between 4000 to 5000 bookings. This 
                variable can have some or no influence towards the predictor. Let the model decide if this needs to be added or not.

      workingday: Almost 69% of the bike booking were happening in ‘workingday’ with a median of close to 5000 booking (for the period of
                  2 years). This indicates, workingday can be a good predictor for the dependent variable

      yr: This shows increase in trend with base year(2018) median below 4000 and subsesquent year as "2019" as median near 6000. This can
                be a good predictor for the dependent variable

- Pair-Plot tells us that there is a Linear relation between 'temp','atemp' and 'cnt

- The heatmap shows some useful insights:

          Correlation of Count('cnt') with independent variables:

          Count('cnt') is highly (positively) correlated with 'casual' and 'registered' and further it is high with 'atemp'. We can
          clearly understand the high positive correlation of count with 'registered' and 'casual' as both of them together add up to 
          represent count.
        
          Count is negatively correlated to 'windspeed' (-0.24 approximately). This gives us an impression that the shared bikes demand
          will be somewhat less on windy days as compared to normal days.

          Correlation among independent variables:
          Some of the independent variables are highly correlated (look at the top-left part of matrix): atemp and temp are highly
          (positively) correlated. The correlation between the two is almost equal to 1.
          Thus, while building the model, we'll have to pay attention to multicollinearity.

          We will refer this map back-and-forth while building the linear model so as to validate different correlated values along with
          VIF & p-value, for identifying the correct variable to select/eliminate from the model.

-    SPLITTING THE DATA
         Splitting the data to Train and Test: - We will now split the data into TRAIN and TEST (70:30 ratio)
         We will use train_test_split method from sklearn package for this

-   Scaling
         Now that we have done the test-train split, we need to scale the variables for better interpretability. But we only need the
         scale the numeric columns and not the dummy variables. Let's take a look at the list of numeric variables we had created in the
         beginning. Also, the scaling has to be done only on the train dataset as you don't want it to learn anything from the test data.

- Initiallly the model is build with all features and combination of RFE and manual deletion of indepedent variables were done in model
        building

-  From the model summary above, all the variables have p-value < 0.05 and from the p-value perspective, all variables seem significant.
   Also have the variables have VIF < 5. There seems to be VERY LOW Multicollinearity between the predictors.Also the Adjusted R-squared
   value has dropped from 84.5% with 28 variables to just 83.0% using 9 variables. The model seems to be doing a good job. 

- Residual Analysis Of Training Data
      - Residuals are normally distributed. Hence our assumption for Linear Regression is valid.
      - VIF values are less than 5 which is good and also there is no multicolinearity as seen from the heatmap.
      - Linearity can be observed from above visualizations.
      - Homoscedasticity -No visible pattern observed from above plot for residuals.
      - Durbin-Watson value of final model lr6 is 1.977, which signifies there is no autocorrelation

- Model is doing well on the test set as well.
      - R^2 Value for TEST
           0.8090249807379293
      - Adjusted R^2 Value for TEST
           r2=0.8203092200749708

- Final model consists of the 9 variables mentioned above.One can go ahead with this model and use it for predicting count of daily bike
  rentals.   

- We can see that the equation of our best fitted line is:

       cnt = 0.2546 + 0.2348*yr + 0.4519*temp -0.1429*windspeed -0.1133*season_spring + 0.0458*season_winter -0.0702*mnth_Jul + 0.0525* 
             mnth_Sept + 0.0154* weekday_Sat -0.0416*weekday_Sun -0.0811*weathersit_B -0.2870*weathersit_C

- Final Result Comparison

         Train R^2 :0.0.833
         Train Adjusted R^2 :0.830
         Test R^2 :808
         Test Adjusted R^2 :0.812
         This seems to be a really good model that can very well 'Generalize' various datasets. 

-As per our final Model, the top 3 predictor variables that influences the bike booking are:
        . Temperature (temp) - A coefficient value of ‘0.4509’ indicated that a unit increase in temp variable increases the bike hire
                               numbers by '0.4509' units.
        . Weather Situation 3 (weathersit_C) - A coefficient value of ‘-0.2868’ indicated that, w.r.t Weathersit1, a unit increase in 
                               Weathersit3 variable decreases the bike hire numbers by '-0.2868' units.
        . Year (yr) - A coefficient value of ‘0.2344’ indicated that a unit increase in yr variable increases the bike hire numbers by
                      '0.2344' units.So, it's suggested to consider these variables utmost importance while planning, to achive maximum
                       booking.


          The next best features that can also be considered are

              . season_winter & mnth_Sept : - A coefficient value of ‘0.0461’ * '0.0526' respectively. This indicates that both are next
                                              best influencer.
              . windspeed: - A coefficient value of ‘-0.1414’ indicated that, a unit increase in windspeed variable decreases the bike
                                              hire numbers by 0.1414 units.
               NOTE: The details of weathersit_B & weathersit_C weathersit_B: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
                      weathersit_C: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds

         - The details of season_spring & season_winter
           season_spring: spring season_winter: winter 


## Technologies Used
- library - version 1.0
- library - version 2.0
- library - version 3.0

## Acknowledgements


## Contact
Created by [@githubusername] - https://github.com/amitp-0457/Bike_Sharing_Assingment

