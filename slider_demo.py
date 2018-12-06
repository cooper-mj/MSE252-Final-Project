import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math

fig, ax = plt.subplots()

plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)

# Defining our starting variables and step sizes for each of our sliders
doppler0 = 0.5 # Initial accuracy of doppler
doppler_step = 0.01

rain0 = 0.5 # Initial probability of rain
rain_step = 0.01 # Step size for rain probability



s = 0 # starting bar heights
l = plt.bar([0, 1], [s, s], color='red')

plt.axis([0, 1, -100000, 100000])
plt.axhline(y=0, color='black')
plt.ylabel("Utility")
plt.xlabel("Wait to Harvest (left); Harvest Immediately (right)")

axcolor = 'lightgoldenrodyellow'
doppler_accuracy = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
prob_rain = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

accuracy = Slider(doppler_accuracy, 'Accuracy of SuperDoppler', 0, 1, valinit=doppler0, valstep=doppler_step)
prain = Slider(prob_rain, 'Probability of Rain', 0, 1, valinit=rain0, valstep=rain_step)


def update(val):
    rain = prain.val
    acc = accuracy.val

    l[0].set_height(utility_neither(rain)[0])
    l[1].set_height(utility_neither(rain)[1])

    fig.canvas.draw_idle()
prain.on_changed(update)
accuracy.on_changed(update)

resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    prain.reset()
    accuracy.reset()
button.on_clicked(reset)

rax = plt.axes([0.0, 0.7, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('Purchase Doppler', 'No Doppler'), active=0)


rax2 = plt.axes([0.0, 0.5, 0.15, 0.15], facecolor=axcolor)
radio2 = RadioButtons(rax2, ('Purchase Spores', 'No Spores'), active=0)

def utility_function(x, rho):
	return math.e**(-float(x)/rho)

def inverse_utility_function(u, rho):
	return -rho*math.log(float(u))

def utility_neither(rain_probability, rho=74844, mold_probability_given_rain=0.4, mold_probability_given_no_rain=0.25, acid_rise_probability=0.9, acid_20_probability=0.5, acid_25_probability=0.5):
	# These four variables represent the value of waiting dependent on
	# certain conditions.
	value_storm_botrytis = utility_function(82200.0, rho)
	value_storm_Nbotrytis = utility_function(12900.0, rho)
	value_Nstorm_botrytis = utility_function(82200.0, rho)
	value_acid_rise_25 = utility_function(42000.0, rho)
	value_acid_rise_20 = utility_function(36000.0, rho)
	value_acid_no_rise = utility_function(30000.0, rho)
	value_Nstorm_Nbotrytis = (acid_rise_probability * acid_25_probability * value_acid_rise_25) + (acid_rise_probability * acid_20_probability * value_acid_rise_20) + (1 - acid_rise_probability * value_acid_no_rise)
	e_value_wait = (rain_probability * mold_probability_given_rain * value_storm_botrytis) + (rain_probability * (1 - mold_probability_given_rain) * value_storm_Nbotrytis) + ((1 - rain_probability) * mold_probability_given_no_rain * value_Nstorm_botrytis) + ((1 - rain_probability) * (1 - mold_probability_given_no_rain) * value_Nstorm_Nbotrytis)

	# This represents the value of harvesting now
	e_value_harvest_now = utility_function(34200.0, rho)

	print([inverse_utility_function(e_value_wait, rho), inverse_utility_function(e_value_harvest_now, rho)])
	return [inverse_utility_function(e_value_wait, rho), inverse_utility_function(e_value_harvest_now, rho)]


def utility_only_spores():
	rain_probability, mold_probability
# def colorfunc(label):
#     l.set_color(label)
#     fig.canvas.draw_idle()
# radio.on_clicked(colorfunc)

plt.show()