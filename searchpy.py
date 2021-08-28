from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:\Windows\Fonts\malgun.ttf").get_name()
rc('font', family=font_name)

import cv2
import numpy as np
class Recognition:
    def ExtractNumber(self, src):
        try:
            img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_UNCHANGED)
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(img2, (3, 3), 0)
            canny = cv2.Canny(blur, 5, 800)
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            box1 = []
            sumarea = 0
            for i in range(len(contours)):
                cnt = contours[i]
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                if (rect_area >= 9) and (rect_area <= 2550):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    sumarea += w * h
                    box1.append(cv2.boundingRect(cnt))
            return sumarea
        except:
            return -1

# 최소편집 알고리즘
def edit_distance(s1, s2):
    l1, l2 = len(s1), len(s2)
    if l2 > l1:
        return edit_distance(s2, s1)
    if l2 is 0:
        return l1
    prev_row = list(range(l2 + 1))
    current_row = [0] * (l2 + 1)
    for i, c1 in enumerate(s1):
        current_row[0] = i + 1
        for j, c2 in enumerate(s2):
            d_ins = current_row[j] + 1
            d_del = prev_row[j + 1] + 1
            d_sub = prev_row[j] + (1 if c1 != c2 else 0)
            current_row[j + 1] = min(d_ins, d_del, d_sub)
        prev_row[:] = current_row[:]
    return prev_row[-1]

def Dataization(img_path):
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dsize=(64, 64))
    return (img / 256)

from flask_server import foodOlist, foodXlist, imgadvX, imgadvO

def estimate(searchname, selection, model1, model2):
    findname = searchname
    if str(selection) == 'hash':
        url = 'https://www.instagram.com/explore/tags/' + str(findname)
    else:
        url = 'https://www.instagram.com/' + str(findname)
    # 기본 함수
    import urllib.request
    from bs4 import BeautifulSoup
    import re
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    import time
    import os

    # 기본데이터
    global foodOlist
    global foodXlist
    global imgadvX
    global imgadvO

    path_dir = 'image/intest/'
    file_list = os.listdir(path_dir)
    for i in range(0, len(file_list)):
        os.remove(path_dir + file_list[i])

    headers = {'User-Agent': 'Chrome/66.0.3359.181'}
    driver = webdriver.Chrome('chromedriver2.exe')
    driver.get(url)
    filenum = 0
    filesrc = []

    time.sleep(5)
    body = driver.find_element_by_xpath('/html/body')
    driver.find_elements_by_xpath("//div[contains(@class,'_9AhH0')]")[0].click()
    time.sleep(1)

    url2 = driver.current_url
    driver2 = webdriver.Chrome('chromedriver2.exe')
    driver2.get(url2)
    html = driver2.page_source

    src = re.findall('www.instagram.com\/p\/(.*?)\/', str(url2))
    if str(src).replace("'", '').replace("]", "").replace("[", "") not in filesrc:
        bsObject = BeautifulSoup(html, "lxml")
        picsrc = re.findall('srcset=\"https(.*?) ', str(bsObject))
        if len(picsrc) != 0:
            picsrc = 'https' + picsrc[0].replace('&amp;', '&')
        if len(picsrc) != 0:
            filesrc.append(str(src).replace('"', '').replace("'", '').replace("]", "").replace("[", ""))
            gpicsrc = str(picsrc).replace('&', '%26').replace('=', '%3D').replace('?', '%3F').replace(':', '%3A').replace(
                '/', '%2F')
            driver2.get('https://www.google.no/searchbyimage?image_url=' + str(
                gpicsrc) + '&encoded_image=&image_content=&filename=&hl=ko')
            time.sleep(0.5)
            html2 = driver2.page_source
            bsObject2 = BeautifulSoup(html2, "lxml")
            k = re.findall('title=\"검색\" type=\"text\" value=\"(.*?)\"\/\>\<\/div\>', str(bsObject2))
            l = str(k).split(',')
            l = str(filenum).zfill(5) + '_' + l[0].replace("'", '').replace("]", "").replace("[", "").replace(" ",
                                                                                                              "") + '_' + str(
                src).replace("'", '').replace("]", "").replace("[", "")
            ur = "image/intest/" + str(l.replace('"', '').replace("'", '').replace("[", "").replace(" ", "")) + ".jpg"
            urllib.request.urlretrieve(str(picsrc).replace('"', '').replace("'", ''), ur)
            filenum += 1

    pgnum = 50
    while pgnum:
        pgnum -= 1
        body.send_keys(Keys.RIGHT)
        time.sleep(1)
        url2 = driver.current_url
        driver2.get(url2)
        html = driver2.page_source
        src = re.findall('www.instagram.com\/p\/(.*?)\/', str(url2))
        if str(src).replace("'", '').replace("]", "").replace("[", "") not in filesrc:
            bsObject = BeautifulSoup(html, "lxml")
            picsrc = re.findall('srcset=\"https(.*?) ', str(bsObject))
            if len(picsrc) != 0:
                picsrc = 'https' + picsrc[0].replace('&amp;', '&')
            if len(picsrc) != 0:
                filesrc.append(str(src).replace('"', '').replace("'", '').replace("]", "").replace("[", ""))
                gpicsrc = str(picsrc).replace('&', '%26').replace('=', '%3D').replace('?', '%3F').replace(':',
                                                                                                          '%3A').replace(
                    '/', '%2F')
                driver2.get('https://www.google.no/searchbyimage?image_url=' + str(
                    gpicsrc) + '&encoded_image=&image_content=&filename=&hl=ko')
                time.sleep(0.5)
                html2 = driver2.page_source
                bsObject2 = BeautifulSoup(html2, "lxml")
                k = re.findall('title=\"검색\" type=\"text\" value=\"(.*?)\"\/\>\<\/div\>', str(bsObject2))
                l = str(k).split(',')
                l = str(filenum).zfill(5) + '_' + l[0].replace("'", '').replace("]", "").replace("[", "").replace(" ",
                                                                                                                  "") + '_' + str(
                    src).replace("'", '').replace("]", "").replace("[", "")
                ur = "image/intest/" + str(l.replace('"', '').replace("'", '').replace("[", "").replace(" ", "")) + ".jpg"
                urllib.request.urlretrieve(str(picsrc).replace('"', '').replace("'", ''), ur)
                filenum += 1
        if filenum >= 15:
            break

    driver2.close()
    driver.close()

    # 인스타 점수 평가


    print1 = []
    print2 = []
    print3 = []

    # 사진속 넓이 반환 함수
    import cv2
    import os
    import numpy as np


    # 파일 이름을 통해 음식인지 아닌지 확인
    path_dir = 'image/intest'
    intestname = []
    file_list = os.listdir(path_dir)
    for i in range(0, len(file_list)):
        intestname.append(file_list[i].split('_')[1])

    for j in range(0, len(file_list)):
        word = intestname[j]
        wordtype = 0
        wordlen = 9999
        for i in range(0, len(foodOlist)):
            if wordlen > edit_distance(word, str(foodOlist[0][i])):
                wordlen = edit_distance(word, str(foodOlist[0][i]))
                wordsim = str(foodOlist[0][i])
                wordtype = 1
        for i in range(0, len(foodXlist)):
            if wordlen > edit_distance(word, str(foodXlist[0][i])):
                wordlen = edit_distance(word, str(foodXlist[0][i]))
                wordsim = str(foodXlist[0][i])
                wordtype = 2
        # 음식이라면 광고확률을 출력해서 평가
        if wordtype == 1:
            recogtest = Recognition()
            result = recogtest.ExtractNumber('image/intest/' + str(file_list[j]))
            if result != -1:
                a = len(imgadvO)
                b = len(imgadvX)
                ac = 1
                bc = 1
                for j in range(0, len(imgadvO)):
                    if (result <= imgadvO[j] + 100) and (result >= imgadvO[j] - 50):
                        ac += 1
                for j in range(0, len(imgadvX)):
                    if (result <= imgadvX[j] + 100) and (result >= imgadvX[j] - 50):
                        bc += 1
                if (ac / a) / (ac / a + bc / b) >= 0.5:
                    print1.append("A : ({:.4f})".format((ac / a) / (ac / a + bc / b)))
                #                 print("광고 O : 불일치도(",wordlen,") 비슷한단어 : ",wordsim,"({:.4f})".format((ac/a)/(ac/a+bc/b)))
                else:
                    print1.append("F : ({:.4f})".format(1 - (ac / a) / (ac / a + bc / b)))
            #                 print("광고 X : 불일치도(",wordlen,") 비슷한단어 : ",wordsim,"({:.4f})".format((ac/a)/(ac/a+bc/b)))
            else:
                print1.append("Error")
        elif wordtype == 2:
            print1.append("N : " + wordsim)
        #         print("음식X : 불일치도(",wordlen,") 비슷한단어 : ",wordsim)
        else:
            print("Error")


    # CNN을 통해 분류한결과 출력 (음식X : N, 음식:F, 광고:A)
    categories = ["A", "F", "N"]

    src = []
    name = []
    test = []


    for file in os.listdir('image/intest'):
        if (file.find('.jpg') is not -1):
            src.append('image/intest/' + file)
            name.append(file)
            test.append(Dataization('image/intest/' + file))

    test = np.array(test)
    predict = model1.predict_classes(test)
    predictions2 = model1.predict(test)

    for i in range(len(test)):
        print2.append(str(categories[predict[i]]))


    # CNN을 통해 좋아요 분류해 ♥로 출력
    categories = ["♥♡♡♡♡", "♥♥♡♡♡", "♥♥♥♡♡", "♥♥♥♥♡", "♥♥♥♥♥"]

    src = []
    name = []
    test = []


    for file in os.listdir('image/intest'):
        if (file.find('.jpg') is not -1):
            src.append('image/intest/' + file)
            name.append(file)
            test.append(Dataization('image/intest/' + file))

    test = np.array(test)
    predict = model2.predict_classes(test)
    predictions = model2.predict(test)

    for i in range(len(test)):
        print3.append(str(categories[predict[i]]))


    import matplotlib.pyplot as plt


    def plot_image(i, predictions_array, predictions2_array, true_label, imag):
        predictions_array, imag = predictions_array[i], imag[i]
        predictions2_array = predictions2_array[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        b, g, r = cv2.split(imag)
        imag = cv2.merge([r, g, b])
        plt.imshow(imag, cmap=plt.cm.binary)
        plt.xlabel(print1[i] + "\n" + print2[i] + " : ({:0.4f})".format(np.max(predictions2_array)))


    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(5), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        plt.xlabel(print3[i] + "\n{:2.2f}%".format(100 * np.max(predictions_array)))


    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, predictions2, categories, test)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test)
    plt.subplots_adjust(hspace=.55)
    plt.savefig('static/result-' + str(findname) + '.jpg')
    return 'result-' + str(findname) + '.jpg'

def urlsearch(imgurl,model1, model2):
    import urllib.request
    from bs4 import BeautifulSoup
    import re
    from selenium import webdriver
    import time
    import os

    # 기본데이터
    global foodOlist
    global foodXlist
    global imgadvX
    global imgadvO

    path_dir = 'image/intest/'
    file_list = os.listdir(path_dir)
    for i in range(0, len(file_list)):
        os.remove(path_dir + file_list[i])

    driver = webdriver.Chrome('chromedriver2.exe')
    driver.get('https://www.google.no/searchbyimage?image_url=' + str(imgurl) + '&encoded_image=&image_content=&filename=&hl=ko')
    time.sleep(0.5)

    html2 = driver.page_source
    bsObject2 = BeautifulSoup(html2, "lxml")
    k = re.findall('title=\"검색\" type=\"text\" value=\"(.*?)\"\/\>\<\/div\>', str(bsObject2))
    l = str(k).split(',')
    l = l[0].replace("'", '').replace("]", "").replace("[", "").replace(" ","")
    ur = "image/intest/" + str(l.replace('"', '').replace("'", '').replace("[", "").replace(" ", "")) + ".jpg"
    urllib.request.urlretrieve(str(imgurl), ur)
    driver.close()


    print1 = []
    print2 = []
    print3 = []

    # 사진속 넓이 반환 함수
    import cv2
    import os
    import numpy as np

    # 파일 이름을 통해 음식인지 아닌지 확인
    path_dir = 'image/intest'
    intestname = []
    file_list = os.listdir(path_dir)
    for i in range(0, len(file_list)):
        intestname.append(file_list[i].split('.')[0])

    for j in range(0, len(file_list)):
        word = intestname[j]
        wordtype = 0
        wordlen = 9999
        for i in range(0, len(foodOlist)):
            if wordlen > edit_distance(word, str(foodOlist[0][i])):
                wordlen = edit_distance(word, str(foodOlist[0][i]))
                wordsim = str(foodOlist[0][i])
                wordtype = 1
        for i in range(0, len(foodXlist)):
            if wordlen > edit_distance(word, str(foodXlist[0][i])):
                wordlen = edit_distance(word, str(foodXlist[0][i]))
                wordsim = str(foodXlist[0][i])
                wordtype = 2
        # 음식이라면 광고확률을 출력해서 평가
        if wordtype == 1:
            recogtest = Recognition()
            result = recogtest.ExtractNumber('image/intest/' + str(file_list[j]))
            if result != -1:
                a = len(imgadvO)
                b = len(imgadvX)
                ac = 1
                bc = 1
                for j in range(0, len(imgadvO)):
                    if (result <= imgadvO[j] + 100) and (result >= imgadvO[j] - 50):
                        ac += 1
                for j in range(0, len(imgadvX)):
                    if (result <= imgadvX[j] + 100) and (result >= imgadvX[j] - 50):
                        bc += 1
                if (ac / a) / (ac / a + bc / b) >= 0.5:
                    print1.append("A : ({:.4f})".format((ac / a) / (ac / a + bc / b)))
                #                 print("광고 O : 불일치도(",wordlen,") 비슷한단어 : ",wordsim,"({:.4f})".format((ac/a)/(ac/a+bc/b)))
                else:
                    print1.append("F : ({:.4f})".format(1 - (ac / a) / (ac / a + bc / b)))
            #                 print("광고 X : 불일치도(",wordlen,") 비슷한단어 : ",wordsim,"({:.4f})".format((ac/a)/(ac/a+bc/b)))
            else:
                print1.append("Error")
        elif wordtype == 2:
            print1.append("N : " + wordsim)
        #         print("음식X : 불일치도(",wordlen,") 비슷한단어 : ",wordsim)
        else:
            print("Error")

    # CNN을 통해 분류한결과 출력 (음식X : N, 음식:F, 광고:A)
    categories = ["A", "F", "N"]

    src = []
    name = []
    test = []


    for file in os.listdir('image/intest'):
        if (file.find('.jpg') is not -1):
            src.append('image/intest/' + file)
            name.append(file)
            test.append(Dataization('image/intest/' + file))

    test = np.array(test)
    predict = model1.predict_classes(test)
    predictions2 = model1.predict(test)

    for i in range(len(test)):
        print2.append(str(categories[predict[i]]))

    # CNN을 통해 좋아요 분류해 ♥로 출력
    categories = ["♥♡♡♡♡", "♥♥♡♡♡", "♥♥♥♡♡", "♥♥♥♥♡", "♥♥♥♥♥"]

    src = []
    name = []
    test = []


    for file in os.listdir('image/intest'):
        if (file.find('.jpg') is not -1):
            src.append('image/intest/' + file)
            name.append(file)
            test.append(Dataization('image/intest/' + file))

    test = np.array(test)
    predict = model2.predict_classes(test)
    predictions = model2.predict(test)

    for i in range(len(test)):
        print3.append(str(categories[predict[i]]))

    import matplotlib.pyplot as plt

    def plot_value_array(i, predictions_array, true_label,predictions2_array):
        predictions_array, true_label = predictions_array[i], true_label[i]
        predictions2_array = predictions2_array[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(5), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        plt.xlabel(print1[i] + "\n" + print2[i] + " : ({:0.4f})".format(np.max(predictions2_array))+"\n"+print3[i] + "\n{:2.2f}%".format(100 * np.max(predictions_array)))

    plt.figure(figsize=(6.4,6.4))
    plt.subplots_adjust(hspace=.55)
    plot_value_array(0, predictions, test,predictions2)
    plt.savefig('static/result-url.jpg')
    return 'result-url.jpg'


def uploadsearch(model1, model2):
    import os

    # 기본데이터
    global foodOlist
    global foodXlist
    global imgadvX
    global imgadvO

    print1 = []
    print2 = []
    print3 = []


    import numpy as np


    # 파일 이름을 통해 음식인지 아닌지 확인
    path_dir = 'static/upload'
    file_list = os.listdir(path_dir)

    for j in range(0, len(file_list)):
        recogtest = Recognition()
        result = recogtest.ExtractNumber('image/intest/' + str(file_list[j]))
        if result != -1:
            a = len(imgadvO)
            b = len(imgadvX)
            ac = 1
            bc = 1
            for j in range(0, len(imgadvO)):
                if (result <= imgadvO[j] + 100) and (result >= imgadvO[j] - 50):
                    ac += 1
            for j in range(0, len(imgadvX)):
                if (result <= imgadvX[j] + 100) and (result >= imgadvX[j] - 50):
                    bc += 1
            if (ac / a) / (ac / a + bc / b) >= 0.5:
                print1.append("A : ({:.4f})".format((ac / a) / (ac / a + bc / b)))
            else:
                print1.append("F : ({:.4f})".format(1 - (ac / a) / (ac / a + bc / b)))
        else:
            print1.append("Error")

    # CNN을 통해 분류한결과 출력 (음식X : N, 음식:F, 광고:A)
    categories = ["A", "F", "N"]

    src = []
    name = []
    test = []


    for file in os.listdir('static/upload'):
        if (file.find('.jpg') is not -1):
            src.append('static/upload/' + file)
            name.append(file)
            test.append(Dataization('static/upload/' + file))

    test = np.array(test)
    predict = model1.predict_classes(test)
    predictions2 = model1.predict(test)

    for i in range(len(test)):
        print2.append(str(categories[predict[i]]))

    # CNN을 통해 좋아요 분류해 ♥로 출력
    categories = ["♥♡♡♡♡", "♥♥♡♡♡", "♥♥♥♡♡", "♥♥♥♥♡", "♥♥♥♥♥"]

    src = []
    name = []
    test = []


    for file in os.listdir('static/upload'):
        if (file.find('.jpg') is not -1):
            src.append('static/upload/' + file)
            name.append(file)
            test.append(Dataization('static/upload/' + file))

    test = np.array(test)
    predict = model2.predict_classes(test)
    predictions = model2.predict(test)

    for i in range(len(test)):
        print3.append(str(categories[predict[i]]))

    import matplotlib.pyplot as plt

    def plot_value_array(i, predictions_array, true_label,predictions2_array):
        predictions_array, true_label = predictions_array[i], true_label[i]
        predictions2_array = predictions2_array[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(5), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        plt.xlabel(print2[i] + " : ({:0.4f})".format(np.max(predictions2_array))+"\n"+print3[i] + "\n{:2.2f}%".format(100 * np.max(predictions_array)))

    plt.figure(figsize=(6.4,6.4))
    plt.subplots_adjust(hspace=.55)
    plot_value_array(0, predictions, test,predictions2)
    plt.savefig('static/result-url.jpg')
    return 'result-url.jpg'

def process(imgurl,model1,model2):

    # 기본데이터
    global foodOlist
    global foodXlist
    global imgadvX
    global imgadvO

    imgname = imgurl.split('_')[3]


    print1 = []
    print2 = []
    print3 = []

    # 사진속 넓이 반환 함수
    import numpy as np


    # 파일 이름을 통해 음식인지 아닌지 확인
    word = imgname
    wordt1 = []
    wordl1 = []
    wordt2 = []
    wordl2 = []
    for i in range(0, len(foodOlist)):
            wordl1.append(edit_distance(word, str(foodOlist[0][i])))
            wordt1.append(str(foodOlist[0][i]))
    for i in range(0, len(foodXlist)):
        wordl2.append(edit_distance(word, str(foodXlist[0][i])))
        wordt2.append(str(foodXlist[0][i]))

    for i in range(0, len(foodOlist)-1):
        for j in range(i, len(foodOlist)):
            if wordl1[i]>wordl1[j]:
                wordl1[j],wordl1[i] = wordl1[i],wordl1[j]
                wordt1[j],wordt1[i] = wordt1[i],wordt1[j]
    for i in range(0, len(foodXlist)-1):
        for j in range(i, len(foodXlist)):
            if wordl2[i]>wordl2[j]:
                wordl2[j],wordl2[i] = wordl2[i],wordl2[j]
                wordt2[j],wordt2[i] = wordt2[i],wordt2[j]

    recogtest = Recognition()
    result = recogtest.ExtractNumber('static/sample/' + str(imgurl))
    if result != -1:
        a = len(imgadvO)
        b = len(imgadvX)
        ac = 1
        bc = 1
        for j in range(0, len(imgadvO)):
            if (result <= imgadvO[j] + 100) and (result >= imgadvO[j] - 50):
                ac += 1
        for j in range(0, len(imgadvX)):
            if (result <= imgadvX[j] + 100) and (result >= imgadvX[j] - 50):
                bc += 1
        if (ac / a) / (ac / a + bc / b) >= 0.5:
            print1.append("광고 : ({:.4f})".format((ac / a) / (ac / a + bc / b)))
        else:
            print1.append("음식 : ({:.4f})".format(1 - (ac / a) / (ac / a + bc / b)))



    categories = ["광고", "음식", "기타"]

    test = []
    test.append(Dataization('static/sample/' + str(imgurl)))

    test = np.array(test)
    predict = model1.predict_classes(test)
    predictions2 = model1.predict(test)

    for i in range(len(test)):
        print2.append(str(categories[predict[i]]))

    # CNN을 통해 좋아요 분류해 ♥로 출력
    categories = ["♥♡♡♡♡", "♥♥♡♡♡", "♥♥♥♡♡", "♥♥♥♥♡", "♥♥♥♥♥"]

    test = []
    test.append(Dataization('static/sample/' + str(imgurl)))

    test = np.array(test)
    predict = model2.predict_classes(test)
    predictions = model2.predict(test)

    for i in range(len(test)):
        print3.append(str(categories[predict[i]]))

    import matplotlib.pyplot as plt

    def plot_value_array(i, predictions_array, true_label,predictions2_array):
        predictions_array, true_label = predictions_array[i], true_label[i]
        predictions2_array = predictions2_array[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(5), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        plt.xlabel(print1[i] + "\n" + print2[i] + " : ({:0.4f})".format(np.max(predictions2_array))+"\n"+print3[i] + "\n{:2.2f}%".format(100 * np.max(predictions_array)))

    plt.figure(figsize=(6.4,6.4))
    plt.subplots_adjust(hspace=.55)
    plot_value_array(0, predictions, test,predictions2)
    plt.savefig('static/result-url.jpg')
    return wordt1, wordl1, wordt2, wordl2, print1, print2, print3