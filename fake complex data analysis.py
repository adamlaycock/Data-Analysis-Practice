import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from datetime import date
import numpy as np

# Reading In The Dataset
data=pd.read_csv('complex_fake_data.csv')
# Modifying Data Types
data['Date of Joining'] = pd.to_datetime(data['Date of Joining'])

# Task 1: Summary Statistics
# # Age
mean_age=data['Age'].mean()
median_age=data['Age'].median()
sd_age=data['Age'].std()
print(f"The mean age of this dataset is {mean_age:.0f} with a standard deviation of {sd_age:.0f}. "
      f"The median age is {median_age:.0f}.")
# # Salary
mean_salary=data['Salary'].mean()
median_salary=data['Salary'].median()
sd_salary=data['Salary'].std()
print(f"The mean salary of this dataset is {mean_salary:.2f} with a standard deviation of {sd_salary:.2f}. "
      f"The median salary is {median_salary:.0f}.")

# Task 2: Data Aggregation
# # Grouped Data
plt.figure()
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
sns.violinplot(
    x='Department',
    y='Age',
    data=data,
    hue='Department', 
    palette='viridis', 
    ax=ax1
)
ax1.set_xlabel('')
ax1.set_title('Age & Salary Distribution by Department')
sns.violinplot(
    x='Department', 
    y='Salary', 
    data=data, 
    hue='Department', 
    palette='viridis', 
    ax=ax2
)
plt.show()
# # Counting Employees
department_counts=data.groupby('Department').size().reset_index(name='Employee Number')
department_counts.columns = ['Department', 'Employee Count']
plt.figure()
fig2=sns.barplot(
    x='Department', 
    y='Employee Count', 
    data=department_counts, 
    hue='Department', 
    palette='viridis'
)
fig2.set_title('Number of Employees per Department')
plt.show()

# Task 3: Date Analysis
# # Join Date Range
latest_index=data['Date of Joining'].idxmax()
earliest_index=data['Date of Joining'].idxmin()

# # Average Tenure
tenure_list=[]
for date in data['Date of Joining']:
    difference = (pd.Timestamp.today().normalize() - date).days
    tenure_list.append(difference)
mean_tenure = np.mean(tenure_list)
mean_tenure_years = mean_tenure / 365.25

print(f"The average employee tenure is approximately {mean_tenure_years:.0f} years. "
      f"The employee with the longest tenure is "
      f"{data.loc[earliest_index, 'Name']} who joined on "
      f"{data.loc[earliest_index, 'Date of Joining'].strftime('%Y-%m-%d')}. "
      f"The employee with the shortest tenure is "
      f"{data.loc[latest_index, 'Name']} who joined on "
      f"{data.loc[latest_index, 'Date of Joining'].strftime('%Y-%m-%d')}.")

# Task 4: Categorical Analysis
# # Remote Proportion
plt.figure()
fig3=plt.pie(data['Remote Work'].value_counts(), labels=['Remote', 'Not Remote'], autopct='%1.1f%%')
plt.title('Employee Remote Work Distribution')
plt.show()
# # Most Common Remote Department
data['Remote Work'] = data['Remote Work'].replace({True: 'Yes', False: 'No'})

plt.figure()
fig4 = sns.catplot(
    data=data, 
    kind='count',
    x='Department',
    hue='Remote Work',
    palette='viridis',
    errorbar=None
)
fig4.set_ylabels('Number of Employees')
fig4.set_xticklabels(fontsize='8')
plt.title('Remote Work Distribution by Department')
plt.show()

print("hello")
# Task 5: Manager Analysis
# Employee Numbers
manager_counts=data.groupby('Manager ID').size().reset_index(name='Employee Number')
manager_counts.columns = ['Manager ID', 'Employee Count']
print(f"The average number of employees under one manager is {manager_counts['Employee Count'].mean():.0f}. "
      f"The highest number of employees under one manager is {manager_counts['Employee Count'].max()}. "
      f"While the fewest number of employees under one manager is {manager_counts['Employee Count'].min()}.")
# # Manager Performance Scores
performance=data.groupby('Manager ID')['Performance Score'].mean().reset_index()
performance.columns = ['Manager ID', 'Average Score']
performance = performance.sort_values(by='Manager ID', ascending=False)
performance['Manager ID'] = performance['Manager ID'].astype(str)
plt.figure(figsize=(20, 9))
fig5=sns.barplot(
    x='Average Score',
    y='Manager ID',
    data=performance,
    hue='Manager ID',
    palette='viridis'
)
plt.yticks(fontsize='5')
plt.show()
# # Highest & Lowest Achievers
highest_index=performance['Average Score'].idxmax()
lowest_index=performance['Average Score'].idxmin()
print(f"The lowest average performance score was {performance.loc[lowest_index, 'Average Score']:.2f}. "
      f"This value was tied to Manager {performance.loc[lowest_index, 'Manager ID']}. "
      f"The highest average performance score was {performance.loc[highest_index, 'Average Score']:.2f}. "
      f"This score was tied to Manager {performance.loc[highest_index, 'Manager ID']}.")

# Task 6: Correlational Analysis
# # Salary Age Correlation by Department
departments = []
for department in data['Department']:
    if department not in departments:
        departments.append(department)
colors = ['red', 'orange', 'purple', 'green', 'blue', 'indigo', 'violet']

fig6, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

for ax, department, color in zip(axes, departments, colors):
    sns.regplot(x='Age',
                y='Salary', 
                data=data[data['Department'] == department], 
                ax=ax, 
                color=color
    )
    ax.legend(title=department)
    ax.set_xlabel('')
    ax.set_ylabel('')

fig6.suptitle('There is no correlation between Age and Salary.', fontsize='30')
fig6.text(0.5, 0.04, 'Age', ha='center', va='center', fontsize='20')
fig6.text(0.04, 0.5, 'Salary', ha='center', va='center', rotation='vertical', fontsize='20')

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()

# # Salary Performance Correlation
plt.figure()
fig7, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

for ax, department, color in zip(axes, departments, colors):
    sns.regplot(x='Performance Score',
                y='Salary', 
                data=data[data['Department'] == department], 
                ax=ax, 
                color=color
    )
    ax.legend(title=department)
    ax.set_xlabel('')
    ax.set_ylabel('')

fig7.suptitle('There is no correlation between Performance Score and Salary.', fontsize='30')
fig7.text(0.5, 0.04, 'Performance Score', ha='center', va='center', fontsize='20')
fig7.text(0.04, 0.5, 'Salary', ha='center', va='center', rotation='vertical', fontsize='20')

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()

# Task 7: Visualisation
# # Age & Salary Histograms
plt.figure()
fig8, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

sns.histplot(x='Age', 
             data=data, 
             ax=ax1
)
sns.histplot(x='Salary', 
             data=data, 
             ax=ax2
)
fig8.suptitle('Company Age & Salary Distributions', fontsize='30')
plt.show()

# # Age Salary Scatterplot
plt.figure()
fig9=sns.scatterplot(x='Age', 
                     y='Salary', 
                     data=data, 
                     hue='Department'
)
plt.show()


print('bananas')
