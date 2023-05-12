# Predicting Cornell Course Enrollment Change During Add/Drop

Group members: Jolene Mei (xm87), Charlie Ruan (cfr54), Joe Ye (ty357)

## Project Overview
A group of undergraduate students advised by Professor David Shmoys (named Scheduling Team below) has been helping the University Registrar at Cornell to schedule final exams via integer programming models. Currently, the scheduling optimization model requires the student-level course enrollment information after the Add/Drop period as input. This data is typically not available until a month after the semester begins. However, if the team could take advantage of the pre-Add/Drop (or pre-enrollment) data instead of waiting until the end of the Add/Drop period, this process could be completed much earlier in advance, even before the semester begins. Knowing students' drop activities allows the University Registrar to adjust the number of opening spots for each course. For instance, if the Registrar takes into account the estimated number of students who will drop the course, they can increase the capacity at the beginning of the semester. Thus, the final number of students can match the capacity, allowing more students to take the course they want.

In this project, we are interested in exploring the following question: given the pre-Add/Drop student enrollment information of a semester, can we predict the post-Add/Drop student enrollment of that semester? Due to the limitation of time and data, we will only focus on predicting the number of students dropping a course. Though predicting the number of extra students adding a course is a key component for predicting the total enrollment changes, it is out of the scope of this project.

## Codebase
Note that due to privacy, we do not include the data in this codebase, meaning that none of the code in this codebase can be ran by anyone except the three team members (who have agreed to terms of privacy and confidentiality). 

Regardless, the major components of the codebase are as follows:
- The results and figures in the paper are obtained by running: 
  - data_visulizations.ipynb (Section II of paper)
  - preliminary_approach.ipynb (Section IV of paper)
  - final_results.ipynb (Section V of paper)
- The data-preprocessing folder contains all the relevant data cleaning and feature engineering code. 
- parameter_tunin.ipynb corresponds to Section V.C of the paper, where the results are saved in the folder tuned_models
