import random as rm
from random import randint

min_num = 3
max_num = 19
num_questions = 20
num_correct = 0

for i in range(1, num_questions + 1):
    num1 = randint(min_num, max_num)
    num2 = randint(min_num, max_num)

    num3 = num1*num2
    #print(num3)

    print('\n', num1,' x ', num2,'\n')
    user_answer = int(input())            #'\n', num1,' x ', num2,'\n')

    if user_answer == num3:
        num_correct = num_correct + 1
        print('\nCorrect\n')
    else:
        print('\nIncorrect\n')

print('\nYou got', num_correct, 'correct out of ', num_questions,'.')
