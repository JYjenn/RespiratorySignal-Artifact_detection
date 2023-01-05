# system packages
import copy
from collections import deque
# 3-party packages
import numpy as np
from matplotlib.widgets import RectangleSelector
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
# custom packages


class bio_signal_marker:
    def __init__(self, ts, signal, signal2, label, fs, wheel_sec = 2, screen_sec = 20):
        self.signal = signal
        self.signal2 = signal2
        self.signal_label = label
        self.ts = ts
        self.fs = fs
        self.wheel_sec = wheel_sec
        self.screen_sec = screen_sec
        self.x1 = None
        self.x2 = None
        self.restore_que = deque()
        ## button
        self.sel_button = None
        ## data modifiy
        self.tmp_data = None

    def run(self):
        ## make figure
        self.fig, (self.ax3, self.ax2, self.ax1) = plt.subplots(nrows=3, sharex=True)
        ## show data
        self.line, = self.ax1.plot(self.ts, self.signal)
        self.line2, =self.ax2.plot(self.ts, self.signal2)
        self.line3, = self.ax3.plot(self.ts, self.signal_label)
        self.ax3.set_ylim(bottom = -1, top = 2)
        ## xlim
        self.xlim_start = 0
        self.xlim_end = int(self.fs * self.screen_sec)
        plt.xlim([self.xlim_start, self.xlim_end])
        ## mouse evnet
        self.mouse_click_event = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.mouse_release_evnet = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.mouse_whell_evnet = self.fig.canvas.mpl_connect('scroll_event', self.on_wheel)
        self.rect_sel = RectangleSelector(self.ax1, self.line_select_callback,
                               drawtype='box', useblit=False, button=[1],
                               minspanx=5, minspany=5, spancoords='pixels',
                               interactive=True)
        ## key event
        self.key_press_event = self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        return 1

    def line_select_callback(self, eclick, erelease):

        self.x1, y1 = eclick.xdata, eclick.ydata
        self.x2, y2 = erelease.xdata, erelease.ydata

        rect = plt.Rectangle((min(self.x1, self.x2), min(y1, y2)), np.abs(self.x1 - self.x2), np.abs(y1 - y2))
        print(self.x1, self.x2)

    def on_click(self, event):
        self.rect_sel.set_visible(True)
        print('click')

    def on_release(self, event):
        if str(event.button) == 'MouseButton.LEFT':
            print('L click')

        elif str(event.button) == 'MouseButton.RIGHT':
            self.rect_sel.set_visible(False)
            self.tmp_data = copy.deepcopy(self.signal[int(self.x1):int(self.x2)])

            self.tmp_data_label = copy.deepcopy(self.signal_label[int(self.x1):int(self.x2)])
            self.signal_label[int(self.x1):int(self.x2)] = 1
            self.line3.set_ydata(self.signal_label)

            restore_history_data = [self.x1, self.x2, self.tmp_data, self.tmp_data_label]
            self.restore_que.append(restore_history_data)

            plt.xlim([self.xlim_start, self.xlim_end])
            self.fig.canvas.draw()
        else:
            pass

    def on_wheel(self, event):
        if event.button == 'up':
            if self.xlim_end >= len(self.signal):
                print('wheel up pass')
            else:
                self.xlim_start += int(self.fs * self.wheel_sec)
                self.xlim_end += int(self.fs * self.wheel_sec)
                plt.xlim([self.xlim_start, self.xlim_end])
                self.fig.canvas.draw()
                print('wheel up')
        elif event.button == 'down':
            if self.xlim_start <= 0:
                print('wheel down pass')
            else:
                self.xlim_start -= int(self.fs * self.wheel_sec)
                self.xlim_end -= int(self.fs * self.wheel_sec)
                plt.xlim([self.xlim_start, self.xlim_end])
                self.fig.canvas.draw()
                print('wheel down')
        else:
            pass

    def key_press(self, event):
        print('press', event.key)
        if str(event.key) == 'ctrl+z':
            if len(self.restore_que) == 0:
                pass
            else:
                restore_history_data = self.restore_que.pop()
                tmp_x1, tmp_x2, tmp_signal, tmp_label = restore_history_data
                self.restore_data(self.line3,tmp_label, self.signal_label, tmp_x1, tmp_x2)
                self.fig.canvas.draw()
                plt.xlim([self.xlim_start, self.xlim_end])
        else:
            pass
        return 1

    def restore_data(self, line, tmp_data, raw_signal, x1, x2):
        raw_signal[int(x1):int(x2)] = tmp_data
        line.set_ydata(raw_signal)
        return 1


if __name__ == '__main__':
    pass
