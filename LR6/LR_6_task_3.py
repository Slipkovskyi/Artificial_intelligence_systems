import pandas as pd

# Исходные данные
data = {
    'Day': ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7', 'Day8', 'Day9', 'Day10', 'Day11', 'Day12', 'Day13', 'Day14'],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Частотная таблица
frequency_table_outlook = df.groupby(['Outlook', 'Play']).size().unstack(fill_value=0)
frequency_table_humidity = df.groupby(['Humidity', 'Play']).size().unstack(fill_value=0)
frequency_table_wind = df.groupby(['Wind', 'Play']).size().unstack(fill_value=0)

print("Частотная таблица Outlook:")
print(frequency_table_outlook)
print("\nЧастотная таблица Humidity:")
print(frequency_table_humidity)
print("\nЧастотная таблица Wind:")
print(frequency_table_wind)

# Вероятности для Outlook
p_sunny_yes = 3 / 9
p_sunny_no = 2 / 5
p_rain_yes = 2 / 9
p_rain_no = 2 / 5
p_overcast_yes = 4 / 9
p_overcast_no = 1 / 5

# Вероятности для Humidity
p_high_yes = 3 / 9
p_high_no = 4 / 5
p_normal_yes = 6 / 9
p_normal_no = 1 / 5

# Вероятности для Wind
p_weak_yes = 6 / 9
p_weak_no = 2 / 5
p_strong_yes = 3 / 9
p_strong_no = 3 / 5

# Общие вероятности Yes и No
p_yes = 9 / 14
p_no = 5 / 14

# Расчет правдоподібності
p_yes_given_conditions = p_rain_yes * p_high_yes * p_weak_yes * p_yes
p_no_given_conditions = p_rain_no * p_high_no * p_weak_no * p_no

# Нормализация значений
p_yes_norm = p_yes_given_conditions / (p_yes_given_conditions + p_no_given_conditions)
p_no_norm = p_no_given_conditions / (p_yes_given_conditions + p_no_given_conditions)

print(f'\nВероятность "Yes": {p_yes_norm}')
print(f'Вероятность "No": {p_no_norm}')
