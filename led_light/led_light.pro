TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

#-------------------------------------------------
#   Chiryshev Iurii <iurii.chiryshev@mail.ru>
#   Какие-то общие настройки для проектов
#
#-------------------------------------------------

HOST_NAME = $$QMAKE_HOST.name
#-------------------------------------------------
#   Chiryshev Iurii <iurii.chiryshev@mail.ru>
#   DESKTOP-QRN46PP - home pc
#   DESKTOP-48BO0EE - work pc
#-------------------------------------------------
win32:contains(HOST_NAME,DESKTOP-QRN46PP|DESKTOP-48BO0EE) {
    message( "Chiryshev Iurii $$HOST_NAME detected (home)")
    #-------------------------------------------------
    #              подключение OpenCV
    #-------------------------------------------------
    OPENCV_DIR = 'D:\lib\opencv-2.4.13\build'
    OPENCV_LIB = $$OPENCV_DIR\x86\vc12\lib
    OPENCV_INCLUDE = $$OPENCV_DIR\include

    INCLUDEPATH += $$OPENCV_INCLUDE


    QMAKE_LIBDIR += $$OPENCV_LIB

    CONFIG(debug, debug|release) {
        LIBS += $$files($$OPENCV_LIB\*2413d.lib)
    } else {
        LIBS += $$files($$OPENCV_LIB\*2413.lib)
    }

}
#-------------------------------------------------
#   Chiryshev Iurii <iurii.chiryshev@mail.ru>
#   sis000307.prosoft.ural.ru
#-------------------------------------------------
win32:contains(HOST_NAME,SIS000307) {
    message( "Chiryshev Iurii prosoft detected" )
    #-------------------------------------------------
    #              подключение OpenCV
    #-------------------------------------------------
    OPENCV_DIR = 'D:\libs\opencv-2.4.13.6\build'
    OPENCV_LIB = $$OPENCV_DIR\x64\vc14\lib
    OPENCV_INCLUDE = $$OPENCV_DIR\include

    INCLUDEPATH += $$OPENCV_INCLUDE


    QMAKE_LIBDIR += $$OPENCV_LIB

    CONFIG(debug, debug|release) {
        LIBS += $$files($$OPENCV_LIB\*2413d.lib)
    } else {
        LIBS += $$files($$OPENCV_LIB\*2413.lib)
    }

}

SOURCES += main.cpp \
    blob.cpp \
    ccl.cpp \
    ccl_sauf.cpp \
    led.cpp

HEADERS += \
    ccl.h \
    ccl_sauf.h \
    led.h
