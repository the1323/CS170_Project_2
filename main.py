# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from copy import copy
from typing import List, Any


import math
import csv
import datetime
import multiprocessing as mp
results = []

def collect_result(result):
    global results
    results.append(result)


def leave_one_out_cross_validation(data, current_set_of_features, feature_to_odd,row):

    feature_to_check = copy(current_set_of_features)
    if feature_to_odd in feature_to_check:
        feature_to_check.remove(feature_to_odd)
    else:
        feature_to_check.append(feature_to_odd)

    number_correctly_classfied = 0
    # small has

    # n^2 where n ==500
    for i in range(row):  # 500 it
        lable_object_to_classify = data[i][0]
        nearest_neighbor_distance = float('inf')
        #print(f'Looping over i, at the {i} location, Class {lable_object_to_classify}')
        for k in range(row):  # 500 it
            if i == k:
                # no check if a NN to self
                continue
            distance = 0
            for j in range(len(feature_to_check)):
                dif = data[i][feature_to_check[j]] - data[k][feature_to_check[j]]
                distance = distance + (dif * dif)

            distance = math.sqrt(distance)
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_label = data[k][0]

        if lable_object_to_classify == nearest_neighbor_label:
            number_correctly_classfied += 1
    accuracy = number_correctly_classfied / row
    print(f'Using feature(s) {feature_to_check}  Accuracy is {round(accuracy * 100, 2)}%')
    res = []
    res.append(accuracy)
    res.append(feature_to_odd)
    return res


def feature_search(data):
    # print('search func')
    best_accuracy = 0
    num_cores = int(mp.cpu_count())

    current_set_of_features = []  # chosen best set feature, when end will have all 10 features
    row = len(data)  # 500
    col = len(data[0])  # 11
    for i in range(1, col):
        # print(f'On the {i} th level of the search tree')
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0
        pool = mp.Pool(num_cores)

        for j in range(1, col):

            if j not in current_set_of_features:  # only add if not already in set
                pool.apply_async(leave_one_out_cross_validation, args=(data, current_set_of_features, j,row), callback=collect_result)

        pool.close()
        pool.join()

        for m in range (len(results)):

            if results[m][0] > best_so_far_accuracy :
                best_so_far_accuracy = results[m][0]
                feature_to_add_at_this_level=results[m][1]

        results.clear()
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f'On level {i}, feature {current_set_of_features} was best, accuracy: {round(best_so_far_accuracy * 100, 2)}%')
        file1 = open("myfile.txt", "a")
        file1.write(f'On level {i}, feature {current_set_of_features} was best, accuracy: {round(best_so_far_accuracy * 100, 2)}%\n')
        file1.close()

        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_set = current_set_of_features.copy()
        else:
            print(f'Warning, Accuracy has decreased! Continuing search in case of local maxima')

    print(f'Finished search! The best feature subset is  {best_set}, which has an accuracy of {round(best_accuracy * 100, 2)}%')
    file1 = open("myfile.txt", "a")
    file1.write(
        f'Finished search! The best feature subset is  {best_set}, which has an accuracy of {round(best_accuracy * 100, 2)}%\n')
    file1.close()

def backward_elimination(data):
    num_cores = int(mp.cpu_count())

    best_accuracy = 0
    row = len(data)  # 500
    col = len(data[0])  # 11

    current_set_of_features = []  # chosen best set feature, when end will have all 10 features
    for i in range(1, col):
        current_set_of_features.append(i)

    for i in range(1, col):
        # print(f'On the {i} th level of the search tree')
        feature_to_remove_at_this_level = []
        best_so_far_accuracy = 0
        pool = mp.Pool(num_cores)
        for j in range(1, col):
            if j in current_set_of_features:  # remove if in set
                pool.apply_async(leave_one_out_cross_validation, args=(data, current_set_of_features, j,row),callback=collect_result)

        pool.close()
        pool.join()
        for m in range (len(results)):
            if results[m][0] > best_so_far_accuracy :
                best_so_far_accuracy = results[m][0]
                feature_to_remove_at_this_level=results[m][1]

        results.clear()
        current_set_of_features.remove(feature_to_remove_at_this_level)
        print(f'On level {i}, feature {current_set_of_features} was best, accuracy: {round(best_so_far_accuracy * 100, 2)}%')
        file1 = open("myfile.txt", "a")
        file1.write(f'On level {i}, feature {current_set_of_features} was best, accuracy: {round(best_so_far_accuracy * 100, 2)}%\n')
        file1.close()
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_set = current_set_of_features.copy()
        else:
            print(f'Warning, Accuracy has decreased! Continuing search in case of local maxima')

    print(f'Finished search! The best feature subset is  {best_set}, which has an accuracy of {round(best_accuracy * 100, 2)}%')
    file1 = open("myfile.txt", "a")
    file1.write(f'Finished search! The best feature subset is  {best_set}, which has an accuracy of {round(best_accuracy * 100, 2)}%\n')
    file1.close()

if __name__ == '__main__':

        mp.freeze_support() #mp enable pyinstaller preventing sub processing from explode

        path = str(input('Enter file path\n'))
        data = []

        #path = r'C:\Users\the\Desktop\CS170\project2\Ver_2_CS170_Fall_2021_SMALL_data__61.txt'
        # read file to dataframe in float

        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t')
            for row in spamreader:
                temp=[]
                for v in row[0].split():
                    temp.append(float(v))
                data.append(temp)

        print(f'Dataset size: col: {len(data[0])} row: {len(data)}')
        print("This PC has: " + str(int(mp.cpu_count())) + " cores")
        choice = int(input("1. Forward Search \n2. Backward Search\n"))
        file1 = open("myfile.txt", "a")
        file1.write(f'search: {choice} path: {path} \n')
        file1.close()

        if choice == 1:
            start_t = datetime.datetime.now()
            print('Running Forward Search ')
            feature_search(data)
        if choice == 2:
            start_t = datetime.datetime.now()
            print('Running Backward Search ')
            backward_elimination(data)


        end_t = datetime.datetime.now()
        elapsed_sec = (end_t - start_t).total_seconds()
        elapsed_min = elapsed_sec // 60
        elapsed_sec = elapsed_sec % 60
        print(path)
        print("Total Time :" "{:.0f}".format(elapsed_min) + " Minutes " + "{:.2f}".format(elapsed_sec) + "sec")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
