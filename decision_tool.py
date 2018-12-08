import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math

global using_spores
using_spores = False
global using_doppler
using_doppler = False

fig, ax = plt.subplots()

plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)

# Defining our starting variables and step sizes for each of our sliders
doppler0 = 0.5 # Initial accuracy of doppler
doppler_step = 0.01

rain0 = 0.5 # Initial probability of rain
rain_step = 0.01 # Step size for rain probability

s = 0 # starting bar heights
l = ax.bar([1, 2, 3, 4], [s, s, s, s], color='red')

plt.axis([0, 5, -100000, 100000])
plt.axhline(y=0, color='black')
plt.ylabel("Expected Return ($)")
plt.xlabel("The Green Option is the Choice of Greatest Utility")
ax.set_xticks([1], ["Hi"])

axcolor = 'lightgoldenrodyellow'
doppler_accuracy = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
prob_rain = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

accuracy = Slider(doppler_accuracy, 'Accuracy of SuperDoppler', 0, 1, valinit=doppler0, valstep=doppler_step)
prain = Slider(prob_rain, 'Probability of Rain', 0, 1, valinit=rain0, valstep=rain_step)


def update(val):
    rain = prain.val
    acc = accuracy.val

    ax.texts = [] # Wipe the text labels from prior adjustments
    labels = []

    # Reset colors of all rectangles in the bar graph - this becomes important later
    for rect in l:
        rect.set_color("b")
    max_index = None

    if using_spores and using_doppler:
        values = utility_both(rain)
        l[0].set_height(values[0])
        l[1].set_height(values[1])
        l[2].set_height(values[2])
        l[3].set_height(values[3])
        max_index = values.index(max(values))

        labels = ["Value of Waiting if Doppler says 'Storm'", "Value of Harvesting if Doppler says 'Storm'", "Value of Waiting if Doppler says 'No Storm'", "Value of Harvesting if Doppler says 'No Storm'"]

    elif using_spores:
        values = utility_only_spores(rain)
        l[0].set_height(values[0])
        l[1].set_height(values[1])
        l[2].set_height(values[2])
        l[3].set_height(0)
        max_index = values.index(max(values))

        labels = ["Value of Waiting and Buying Spores", "Value of Waiting and Not Buying Spores", "Value of Harvesting", ""]


    elif using_doppler:
        values = utility_doppler(rain)
        l[0].set_height(values[0])
        l[1].set_height(values[1])
        l[2].set_height(values[2])
        l[3].set_height(values[3])
        max_index = values.index(max(values))

        labels = ["Value of Waiting if Doppler says 'Storm'", "Value of Harvesting if Doppler says 'Storm'", "Value of Waiting if Doppler says 'No Storm'", "Value of Harvesting if Doppler says 'No Storm'"]


    else:
        values = utility_neither(rain)
        l[0].set_height(values[0])
        l[1].set_height(values[1])
        l[2].set_height(0)
        l[3].set_height(0)
        max_index = values.index(max(values))

        labels = ["Value of Waiting", "Value of Harvesting", "", ""]


    # Set color of maximum bar green
    l[max_index].set_color("g")

    # Update labels
    for index, rect in enumerate(l):
        text = labels[index]
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                text,
                ha='center', va='bottom', fontsize=5)

    fig.canvas.draw_idle()

prain.on_changed(update)
accuracy.on_changed(update)

resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def spores_function(label):
    global using_spores

    if label == 'Purchase Spores':
        using_spores = True
    else:
        using_spores = False
    update(prain.val)

def doppler_function(label):
    global using_doppler

    if label == 'Purchase Doppler':
        using_doppler = True
    else:
        using_doppler = False
    update(prain.val)

def reset(event):
    prain.reset()
    accuracy.reset()
button.on_clicked(reset)

rax = plt.axes([0.0, 0.7, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('Purchase Doppler', 'No Doppler'), active=1)
radio.on_clicked(doppler_function)

rax2 = plt.axes([0.0, 0.5, 0.15, 0.15], facecolor=axcolor)
radio2 = RadioButtons(rax2, ('Purchase Spores', 'No Spores'), active=1)
radio2.on_clicked(spores_function)


def utility_function(x, rho):
    return -math.e**(-float(x)/rho)

def inverse_utility_function(u, rho):
    return -rho*math.log(-float(u))

def utility_neither(rain_probability, rho=74844, mold_probability_given_rain=0.4, mold_probability_given_no_rain=0.25, acid_rise_probability=0.8, acid_20_probability=0.5, acid_25_probability=0.5):
    # These four variables represent the value of waiting dependent on
    # certain conditions.

    value_storm_botrytis = utility_function(82200.0, rho)
    value_storm_Nbotrytis = utility_function(12900.0, rho)
    value_Nstorm_botrytis = utility_function(82200.0, rho)
    value_acid_rise_25 = utility_function(42000.0, rho)
    value_acid_rise_20 = utility_function(36000.0, rho)
    value_acid_no_rise = utility_function(30000.0, rho)

    value_acidity_rises = acid_rise_probability*((acid_25_probability * value_acid_rise_25) + (acid_20_probability * value_acid_rise_20))
    value_acididity_no_rise = ((1-acid_rise_probability) * value_acid_no_rise)
    value_Nstorm_Nbotrytis = value_acidity_rises + value_acididity_no_rise

    e_value_wait = (rain_probability * mold_probability_given_rain * value_storm_botrytis) + (rain_probability * (1 - mold_probability_given_rain) * value_storm_Nbotrytis) + ((1 - rain_probability) * mold_probability_given_no_rain * value_Nstorm_botrytis) + ((1 - rain_probability) * (1 - mold_probability_given_no_rain) * value_Nstorm_Nbotrytis)

    # This represents the value of harvesting now
    e_value_harvest_now = utility_function(34200.0, rho)
    return [inverse_utility_function(e_value_wait, rho), inverse_utility_function(e_value_harvest_now, rho)]


def utility_only_spores(rain_probability, rho=74844, mold_probability_given_rain=0.4, mold_probability_given_no_rain=0.25, acid_rise_probability=0.8, acid_20_probability=0.5, acid_25_probability=0.5):
    
    # End up with a list of three values
    # (1) E-value with spores (and waiting)
    # (2) E-value no spores (and waiting)
    # (3) E-value harvesting now (no spores)
    e_value_harvest_now = utility_function(34200.0, rho)

    # Here we calculate the e-value of buying spores
    mold_probability_given_rain_with_spores = 0.999
    spores_storm_mold = rain_probability * mold_probability_given_rain_with_spores * utility_function(71200, rho)
    spores_storm_noMold = rain_probability * (1-mold_probability_given_rain_with_spores) * utility_function(1900, rho)
    spores_noStorm_mold = (1-rain_probability) * mold_probability_given_no_rain * utility_function(71200, rho)
    spores_noStorm_noMold_acidRises_25 = (1-rain_probability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_25_probability * utility_function(31000, rho)
    spores_noStorm_noMold_acidRises_20 = (1-rain_probability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_20_probability * utility_function(25000, rho)
    spores_noStorm_noMold_noAcidRises = (1-rain_probability) * (1-mold_probability_given_no_rain) * (1-acid_rise_probability) * utility_function(19000, rho)
    e_value_spores = spores_storm_mold + spores_storm_noMold + spores_noStorm_mold + spores_noStorm_noMold_acidRises_25 + spores_noStorm_noMold_acidRises_20 + spores_noStorm_noMold_noAcidRises
    

    # Here we calculate the e-value of not buying spores
    noSpores_storm_mold = rain_probability * mold_probability_given_rain * utility_function(81200, rho)
    noSpores_storm_noMold = rain_probability * (1-mold_probability_given_rain) * utility_function(11900, rho)
    noSpores_noStorm_mold = (1-rain_probability) * mold_probability_given_no_rain * utility_function(81200, rho)
    noSpores_noStorm_noMold_acidRises25 = (1-rain_probability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_25_probability * utility_function(41000, rho)
    noSpores_noStorm_noMold_acidRises20 = (1-rain_probability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_20_probability * utility_function(35000, rho)
    noSpores_noStorm_noMold_noAcidRises = (1-rain_probability) * (1-mold_probability_given_no_rain) * (1-acid_rise_probability) * utility_function(29000, rho)
    e_value_noSpores = noSpores_storm_mold + noSpores_storm_noMold + noSpores_noStorm_mold + noSpores_noStorm_noMold_acidRises25 + noSpores_noStorm_noMold_acidRises20 + noSpores_noStorm_noMold_noAcidRises

    return [inverse_utility_function(e_value_spores, rho), inverse_utility_function(e_value_noSpores, rho), inverse_utility_function(e_value_harvest_now, rho)]


def utility_doppler(rain_probability, rho=74844, mold_probability_given_rain=0.4, mold_probability_given_no_rain=0.25, acid_rise_probability=0.8, acid_20_probability=0.5, acid_25_probability=0.5):
    # End up with a list of values
    # (1) If it says "storm" - value of waiting
    # (2) If it says "storm" - value of harvesting now
    # (3) If it says "no storm" - value of waiting
    # (4) If it says "no storm" - value of harvesting now
    
    # Here we calculate (1) - the value of waiting if it says "storm"
    p_Doppler_right = (float(accuracy.val) * prain.val) + ((1-accuracy.val) * (1-prain.val))

    saysStorm_stormProbability = accuracy.val * prain.val / float(p_Doppler_right)
    saysStorm_storm_mold = saysStorm_stormProbability * mold_probability_given_rain * utility_function(81200.0, rho)
    saysStorm_storm_noMold = saysStorm_stormProbability * (1-mold_probability_given_rain) * utility_function(11900.0, rho)
    saysStorm_noStorm_mold = (1-saysStorm_stormProbability) * mold_probability_given_no_rain * utility_function(81200.0, rho)
    saysStorm_noStorm_noMold_acidRises_25 = (1-saysStorm_stormProbability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_25_probability * utility_function(41000.0, rho)
    saysStorm_noStorm_noMold_acidRises_20 = (1-saysStorm_stormProbability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_20_probability * utility_function(35000.0, rho)
    saysStorm_noStorm_noMold_noAcidRises = (1-saysStorm_stormProbability) * (1-mold_probability_given_no_rain) * (1-acid_rise_probability) * utility_function(29000.0, rho)
    e_value_of_waiting_if_saysStorm = saysStorm_storm_mold + saysStorm_storm_noMold + saysStorm_noStorm_mold + saysStorm_noStorm_noMold_acidRises_25 + saysStorm_noStorm_noMold_acidRises_20 + saysStorm_noStorm_noMold_noAcidRises

    # Here we calculate (2) - the value of harvesting now if it says "storm"
    e_value_of_harvesting_if_saysStorm = utility_function(33200.0, rho)

    # Here we calculate (3) - the value of waiting if it says "no storm"
    p_Doppler_wrong = 1-p_Doppler_right
    saysNoStorm_stormProbability = 1-((1-prain.val) * accuracy.val / float(p_Doppler_wrong))

    saysNoStorm_storm_mold = saysNoStorm_stormProbability * mold_probability_given_rain * utility_function(81200.0, rho)
    saysNoStorm_storm_noMold = saysNoStorm_stormProbability * (1-mold_probability_given_rain) * utility_function(11900.0, rho)
    saysNoStorm_noStorm_noMold = (1-saysNoStorm_stormProbability) * mold_probability_given_no_rain * utility_function(81200.0, rho)
    saysNoStorm_noStorm_noMold_acidRises_25 = (1-saysNoStorm_stormProbability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_25_probability * utility_function(41000.0, rho)
    saysNoStorm_noStorm_noMold_acidRises_20 = (1-saysNoStorm_stormProbability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_20_probability * utility_function(35000.0, rho)
    saysNoStorm_noStorm_noMold_noAcidRises = (1-saysNoStorm_stormProbability) * (1-mold_probability_given_no_rain) * (1-acid_rise_probability) * utility_function(29000.0, rho)
    e_value_of_waiting_if_saysNoStorm = saysNoStorm_storm_mold + saysNoStorm_storm_noMold + saysNoStorm_noStorm_noMold + saysNoStorm_noStorm_noMold_acidRises_25 + saysNoStorm_noStorm_noMold_acidRises_20 + saysNoStorm_noStorm_noMold_noAcidRises

    # Here we calculate (4) - the value of harvesting now if it says "no storm"
    e_value_of_harvesting_if_saysNoStorm = utility_function(33200.0, rho)

    return [inverse_utility_function(e_value_of_waiting_if_saysStorm, rho), inverse_utility_function(e_value_of_harvesting_if_saysStorm, rho), inverse_utility_function(e_value_of_waiting_if_saysNoStorm, rho), inverse_utility_function(e_value_of_harvesting_if_saysNoStorm, rho)]

def utility_both(rain_probability, rho=74844, mold_probability_given_rain=0.4, mold_probability_given_no_rain=0.25, acid_rise_probability=0.8, acid_20_probability=0.5, acid_25_probability=0.5):
    # End up with a list of values
    # (1) If it says "storm" and we are using spores - value of waiting
    # (2) If it says "storm" and we are using spores - value of harvesting immediately
    # (3) If it says "no storm" and we are using spores - value of waiting
    # (4) If it says "no storm" and we are using spores - value of harvesting immediately
    mold_probability_given_rain_with_spores = 0.999

    p_Doppler_right = (float(accuracy.val) * prain.val) + ((1-accuracy.val) * (1-prain.val))
    saysStorm_stormProbability = accuracy.val * prain.val / float(p_Doppler_right)
    # Here we calculate (1) - the value of waiting if it says "storm" and we are using spores
    saysStorm_stormProbability = accuracy.val * prain.val / float(p_Doppler_right)
    saysStorm_storm_mold = saysStorm_stormProbability * mold_probability_given_rain_with_spores * utility_function(71200.0, rho)
    saysStorm_storm_noMold = saysStorm_stormProbability * (1-mold_probability_given_rain_with_spores) * utility_function(1900.0, rho)
    saysStorm_noStorm_mold = (1-saysStorm_stormProbability) * mold_probability_given_no_rain * utility_function(71200.0, rho)
    saysStorm_noStorm_noMold_acidRises_25 = (1-saysStorm_stormProbability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_25_probability * utility_function(31000.0, rho)
    saysStorm_noStorm_noMold_acidRises_20 = (1-saysStorm_stormProbability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_20_probability * utility_function(25000.0, rho)
    saysStorm_noStorm_noMold_noAcidRises = (1-saysStorm_stormProbability) * (1-mold_probability_given_no_rain) * (1-acid_rise_probability) * utility_function(19000.0, rho)
    e_value_of_waiting_if_saysStorm_with_spores = saysStorm_storm_mold + saysStorm_storm_noMold + saysStorm_noStorm_mold + saysStorm_noStorm_noMold_acidRises_25 + saysStorm_noStorm_noMold_acidRises_20 + saysStorm_noStorm_noMold_noAcidRises

    # Here we calculate (2) - the value of harvesting immediately if it says "storm" and we are using spores
    e_value_of_harvesting_if_saysStorm_with_spores = utility_function(33200.0, rho)

    saysNoStorm_stormProbability = (1-accuracy.val) * prain.val / float(1-p_Doppler_right)

    # Here we calculate (3) - the value of waiting if it says "no storm" and we are using spores
    saysNoStorm_storm_mold = saysNoStorm_stormProbability * mold_probability_given_rain_with_spores * utility_function(71200.0, rho)
    saysNoStorm_storm_noMold = saysNoStorm_stormProbability * (1-mold_probability_given_rain_with_spores) * utility_function(1900.0, rho)
    saysNoStorm_noStorm_noMold = (1-saysNoStorm_stormProbability) * mold_probability_given_no_rain * utility_function(71200.0, rho)
    saysNoStorm_noStorm_noMold_acidRises_25 = (1-saysNoStorm_stormProbability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_25_probability * utility_function(31000.0, rho)
    saysNoStorm_noStorm_noMold_acidRises_20 = (1-saysNoStorm_stormProbability) * (1-mold_probability_given_no_rain) * acid_rise_probability * acid_20_probability * utility_function(25000.0, rho)
    saysNoStorm_noStorm_noMold_noAcidRises = (1-saysNoStorm_stormProbability) * (1-mold_probability_given_no_rain) * (1-acid_rise_probability) * utility_function(19000.0, rho)
    e_value_of_waiting_if_saysNoStorm_with_spores = saysNoStorm_storm_mold + saysNoStorm_storm_noMold + saysNoStorm_noStorm_noMold + saysNoStorm_noStorm_noMold_acidRises_25 + saysNoStorm_noStorm_noMold_acidRises_20 + saysNoStorm_noStorm_noMold_noAcidRises

    # Here we calculate (4) - the value of harvesting immediately if it says "no storm" and we are using spores
    e_value_of_harvesting_if_saysNoStorm_with_spores = utility_function(33200, rho)

    return [inverse_utility_function(e_value_of_waiting_if_saysStorm_with_spores, rho), inverse_utility_function(e_value_of_harvesting_if_saysStorm_with_spores, rho), inverse_utility_function(e_value_of_waiting_if_saysNoStorm_with_spores, rho), inverse_utility_function(e_value_of_harvesting_if_saysNoStorm_with_spores, rho)]

plt.show()
# Bokeh
