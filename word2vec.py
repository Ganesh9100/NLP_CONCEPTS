#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:55:58 2020

@author: ganesh
"""

import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """
Ambox rewrite.svg	
This article may be in need of reorganization to comply with Wikipedia's layout guidelines. Please help by editing the article to make improvements to the overall structure. (March 2018) (Learn how and when to remove this template message)

Waymo Chrysler Pacifica Hybrid undergoing testing in the San Francisco Bay Area

Autonomous racing car on display at the 2017 New York City ePrix
A self-driving car, also known as an autonomous vehicle (AV), connected and autonomous vehicle (CAV), driverless car, robo-car, or robotic car,[1][2][3] is a vehicle that is capable of sensing its environment and moving safely with little or no human input.[1][4]

Self-driving cars combine a variety of sensors to perceive their surroundings, such as radar, lidar, sonar, GPS, odometry and inertial measurement units.[1] Advanced control systems interpret sensory information to identify appropriate navigation paths, as well as obstacles and relevant signage.[5][6][7]

Long-distance trucking is seen as being at the forefront of adopting and implementing the technology.[8]


Contents
1	History
2	Definitions
2.1	Terminology and safety considerations
2.2	Autonomous vs. automated
2.3	Autonomous versus cooperative
2.4	Self-driving car
2.5	Classification
3	Legal definition
4	Semi-automated vehicles
5	Technical challenges
6	Nature of the digital technology
6.1	Homogenization and decoupling
6.2	Connectivity
6.3	Reprogrammable
6.4	Digital traces
6.5	Modularity
7	Human factor challenges
8	Testing
9	Fields of application
9.1	Autonomous trucks and vans
9.2	Transport systems
10	Impact
10.1	Potential advantages
10.2	Potential disadvantages
11	Potential limits or obstacles
12	Potential changes for different industries
12.1	Taxis
12.2	Healthcare, car repair, and car insurance
12.3	Rescue, emergency response, and military
12.4	Interior design and entertainment
12.5	Telecommunication and energy
12.6	Restaurant, hotels, and airlines
12.7	Elderly and disabled
12.8	Children
13	Incidents
13.1	Tesla Autopilot
13.2	Waymo
13.3	Uber
13.4	Navya automated bus driving system
14	Policy implications
14.1	Urban planning
14.2	Legislation
14.3	Liability
15	Vehicle communication systems
16	Public opinion surveys
17	Moral issues
18	Anticipated launch of cars
19	In fiction
19.1	In film
19.2	In literature
19.3	In television
20	See also
21	References
22	Further reading
History
Main article: History of self-driving cars
Experiments have been conducted on automated driving systems (ADS) since at least the 1920s;[9] trials began in the 1950s. The first semi-automated car was developed in 1977, by Japan's Tsukuba Mechanical Engineering Laboratory, which required specially marked streets that were interpreted by two cameras on the vehicle and an analog computer. The vehicle reached speeds up to 30 kilometres per hour (19 mph) with the support of an elevated rail.[10][11]

A landmark autonomous car appeared in the 1980s, with Carnegie Mellon University's Navlab[12] and ALV[13][14] projects funded by the United States' Defense Advanced Research Projects Agency (DARPA) starting in 1984 and Mercedes-Benz and Bundeswehr University Munich's EUREKA Prometheus Project in 1987.[15] By 1985, the ALV had demonstrated self-driving speeds on two-lane roads of 31 kilometres per hour (19 mph), with obstacle avoidance added in 1986, and off-road driving in day and nighttime conditions by 1987.[16] A major milestone was achieved in 1995, with CMU's NavLab 5 completing the first autonomous coast-to-coast drive of the United States. Of the 2,849 mi (4,585 km) between Pittsburgh, Pennsylvania and San Diego, California, 2,797 mi (4,501 km) were autonomous (98.2%), completed with an average speed of 63.8 mph (102.7 km/h).[17][18][19][20] From the 1960s through the second DARPA Grand Challenge in 2005, automated vehicle research in the United States was primarily funded by DARPA, the US Army, and the US Navy, yielding incremental advances in speeds, driving competence in more complex conditions, controls, and sensor systems.[21] Companies and research organizations have developed prototypes.[15][22][23][24][25][26][27][28][29]

The US allocated US$650 million in 1991 for research on the National Automated Highway System, which demonstrated automated driving through a combination of automation embedded in the highway with automated technology in vehicles, and cooperative networking between the vehicles and with the highway infrastructure. The program concluded with a successful demonstration in 1997 but without clear direction or funding to implement the system on a larger scale.[30] Partly funded by the National Automated Highway System and DARPA, the Carnegie Mellon University Navlab drove 4,584 kilometres (2,848 mi) across America in 1995, 4,501 kilometres (2,797 mi) or 98% of it autonomously.[31] Navlab's record achievement stood unmatched for two decades until 2015, when Delphi improved it by piloting an Audi, augmented with Delphi technology, over 5,472 kilometres (3,400 mi) through 15 states while remaining in self-driving mode 99% of the time.[32] In 2015, the US states of Nevada, Florida, California, Virginia, and Michigan, together with Washington, DC, allowed the testing of automated cars on public roads.[33]

From 2016 to 2018, the European Commission funded an innovation strategy development for connected and automated driving through the Coordination Actions CARTRE and SCOUT.[34] Moreover, the Strategic Transport Research and Innovation Agenda (STRIA) Roadmap for Connected and Automated Transport was published in 2019.[35]

In 2017, Audi stated that its latest A8 would be automated at speeds of up to 60 kilometres per hour (37 mph) using its "Audi AI". The driver would not have to do safety checks such as frequently gripping the steering wheel. The Audi A8 was claimed to be the first production car to reach Level 3 automated driving, and Audi would be the first manufacturer to use laser scanners in addition to cameras and ultrasonic sensors for their system.[36]

In November 2017, Waymo announced that it had begun testing driverless cars without a safety driver in the driver position;[37] however, there was still an employee in the car.[38] In October 2018, Waymo announced that its test vehicles had traveled in automated mode for over 10,000,000 miles (16,000,000 km), increasing by about 1,000,000 miles (1,600,000 kilometres) per month.[39] In December 2018, Waymo was the first to commercialize a fully autonomous taxi service in the US, in Phoenix, Arizona.[40]

A*STAR's Institute for Infocomm Research (I2R) developed a self-driving vehicle which was the first to be approved in Singapore for public road testing at one-north in July 2015. It has ferried several dignitaries such as Prime Minister Lee Hsien Loong, Minister S. Iswaran, Minister Vivian Balakrishnan, and several ministers from other countries.[41][42]

In 2020, a National Transportation Safety Board chairman clarified there is no self-driving car in the US in 2020:

There is not a vehicle currently available to US consumers that is self-driving. Period. Every vehicle sold to US consumers still requires the driver to be actively engaged in the driving task, even when advanced driver assistance systems are activated. If you are selling a car with an advanced driver assistance system, you’re not selling a self-driving car. If you are driving a car with an advanced driver assistance system, you don’t own a self-driving car[43]

Definitions
There is some inconsistency in the terminology used in the self-driving car industry. Various organizations have proposed to define an accurate and consistent vocabulary.

Such confusion has been documented in SAE J3016 which states that "Some vernacular usages associate autonomous specifically with full driving automation (Level 5), while other usages apply it to all levels of driving automation, and some state legislation has defined it to correspond approximately to any ADS [automated driving system] at or above Level 3 (or to any vehicle equipped with such an ADS)."

Terminology and safety considerations
Modern vehicles provide partly automated features such as keeping the car within its lane, speed controls or emergency braking. Nonetheless, differences remain between a fully autonomous self-driving car on one hand and driver assistance technologies on the other hand. According to the BBC, confusion between those concepts leads to deaths.[44]

The Association of British Insurers considers the usage of the word autonomous in marketing for modern cars to be dangerous because car ads make motorists think 'autonomous' and 'autopilot' means a vehicle can drive itself when they still rely on the driver to ensure safety. Technology alone still is not able to drive the car.

When some car makers suggest or claim vehicles are self-driving, when they are only partly automated, drivers risk becoming excessively confident, leading to crashes, while fully self-driving cars are still a long way off in the UK.[45]

Autonomous vs. automated
Autonomous means self-governing.[46] Many historical projects related to vehicle automation have been automated (made automatic) subject to a heavy reliance on artificial aids in their environment, such as magnetic strips. Autonomous control implies satisfactory performance under significant uncertainties in the environment and the ability to compensate for system failures without external intervention.[46]

One approach is to implement communication networks both in the immediate vicinity (for collision avoidance) and farther away (for congestion management). Such outside influences in the decision process reduce an individual vehicle's autonomy, while still not requiring human intervention.

Wood et al. (2012) wrote, "This Article generally uses the term 'autonomous,' instead of the term 'automated.' " The term "autonomous" was chosen "because it is the term that is currently in more widespread use (and thus is more familiar to the general public). However, the latter term is arguably more accurate. 'Automated' connotes control or operation by a machine, while 'autonomous' connotes acting alone or independently. Most of the vehicle concepts (that we are currently aware of) have a person in the driver's seat, utilize a communication connection to the Cloud or other vehicles, and do not independently select either destinations or routes for reaching them. Thus, the term 'automated' would more accurately describe these vehicle concepts."[47] As of 2017, most commercial projects focused on automated vehicles that did not communicate with other vehicles or with an enveloping management regime. EuroNCAP defines autonomous in "Autonomous Emergency Braking" as: "the system acts independently of the driver to avoid or mitigate the accident." which implies the autonomous system is not the driver.[48]

Nonetheless, the words automated and autonomous might also be used together. For instance, Regulation (EU) 2019/2144 of the European Parliament and of the Council of 27 November 2019 on type-approval requirements for motor vehicles (...) defines "automated vehicle" and "fully automated vehicle" based on their autonomous capacity:[49]

"automated vehicle" means a motor vehicle designed and constructed to move autonomously for certain periods of time without continuous driver supervision but in respect of which driver intervention is still expected or required;[49]
"fully automated vehicle" means a motor vehicle that has been designed and constructed to move autonomously without any driver supervision;[49]
Autonomous versus cooperative
To enable a car to travel without any driver embedded within the vehicle, some companies use a remote driver.[citation needed]

According to SAE J3016,
Some driving automation systems may indeed be autonomous if they perform all of their functions independently and self-sufficiently, but if they depend on communication and/or cooperation with outside entities, they should be considered cooperative rather than autonomous.

Self-driving car
PC Magazine defines a self-driving car as "A computer-controlled car that drives itself."[50] The Union of Concerned Scientists states that self-driving cars are "cars or trucks in which human drivers are never required to take control to safely operate the vehicle. Also known as autonomous or 'driverless' cars, they combine sensors and software to control, navigate, and drive the vehicle."[51]

Classification

Tesla Autopilot system is classified as an SAE Level 2 system[52]
A classification system with six levels – ranging from fully manual to fully automated systems – was published in 2014 by SAE International, an automotive standardization body, as J3016, Taxonomy and Definitions for Terms Related to On-Road Motor Vehicle Automated Driving Systems.[53][54] This classification is based on the amount of driver intervention and attentiveness required, rather than the vehicle's capabilities, although these are loosely related. In the United States in 2013, the National Highway Traffic Safety Administration (NHTSA) released a formal classification system,[55] but abandoned it in favor of the SAE standard in 2016. Also in 2016, SAE updated its classification, called J3016_201609.[56]


Levels of driving automation
In SAE's automation level definitions, "driving mode" means "a type of driving scenario with characteristic dynamic driving task requirements (e.g., expressway merging, high speed cruising, low speed traffic jam, closed-campus operations, etc.)"[1][57]

Level 0: The automated system issues warnings and may momentarily intervene but has no sustained vehicle control.
Level 1 ("hands on"): The driver and the automated system share control of the vehicle. Examples are systems where the driver controls steering and the automated system controls engine power to maintain a set speed (Cruise Control) or engine and brake power to maintain and vary speed (Adaptive Cruise Control or ACC); and Parking Assistance, where steering is automated while speed is under manual control. The driver must be ready to retake full control at any time. Lane Keeping Assistance (LKA) Type II is a further example of Level 1 self-driving. A collision mitigation system which alerts the driver to a crash and permits full braking capacity is also a Level 1 feature, according to Autopilot Review magazine.[58]
Level 2 ("hands off"): The automated system takes full control of the vehicle: accelerating, braking, and steering. The driver must monitor the driving and be prepared to intervene immediately at any time if the automated system fails to respond properly. The shorthand "hands off" is not meant to be taken literally – contact between hand and wheel is often mandatory during SAE 2 driving, to confirm that the driver is ready to intervene. The eyes of the driver might be monitored by cameras to confirm that the driver is keeping his/her attention to traffic.
Level 3 ("eyes off"): The driver can safely turn their attention away from the driving tasks, e.g. the driver can text or watch a movie. The vehicle will handle situations that call for an immediate response, like emergency braking. The driver must still be prepared to intervene within some limited time, specified by the manufacturer, when called upon by the vehicle to do so. You can think of the automated system as a co-driver that will alert you in an orderly fashion when it is your turn to drive. An example would be a Traffic Jam Chauffeur.[59]
Level 4 ("mind off"): As level 3, but no driver attention is ever required for safety, e.g. the driver may safely go to sleep or leave the driver's seat. Self-driving is supported only in limited spatial areas (geofenced) or under special circumstances. Outside of these areas or circumstances, the vehicle must be able to safely abort the trip, e.g. park the car, if the driver does not retake control. An example would be a robotic taxi or a robotic delivery service that only covers selected locations in a specific area.
Level 5 ("steering wheel optional"): No human intervention is required at all. An example would be a robotic taxi that works on all roads all over the world, all year around, in all weather conditions.
In the formal SAE definition below, note in particular the shift from SAE 2 to SAE 3: the human driver no longer has to monitor the environment. This is the final aspect of the "dynamic driving task" that is now passed over from the human to the automated system. At SAE 3, the human driver still has responsibility to intervene when asked to do so by the automated system. At SAE 4 the human driver is always relieved of that responsibility and at SAE 5 the automated system will never need to ask for an intervention.
"""



# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)# converting to sentences 

wordss = [nltk.word_tokenize(sentence) for sentence in sentences] # converting the sentence to words 

for i in range(len(wordss)):
    wordss[i] = [word for word in wordss[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(wordss, min_count=2)# this min count is if the words is repeated less than 2 times then that will be removed


words = model.wv.vocab# this is the vocab that we will be working on 

# Finding Word Vectors
#vector = model.wv['love']# default is 100 dimensionaal 

# Most similar words
similar = model.wv.most_similar('drive')# built in fumnc most similar words related to similar 