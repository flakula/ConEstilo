import json
import os
import re

input_folder = "backups/all-content/0.4/"
output_folder = "data/corpus/"

d = {}

save = None

for name in os.listdir(input_folder):
	if name.endswith("txt"):
		# print(name)
		with open(input_folder+name, encoding="utf-8") as f:
			texto = f.read()

		s = re.split(r'(\n\n.*\n.+\n\[.*\]\n\n)', texto)
		h = re.findall(r'(\n\n.*\n.+\n\[.*\]\n\n)', texto)

		temp = [0 , 0]

		for x in range(1, len(s)):
			if x%2==0:
				temp[1] = s[x]
				l = temp[0].find("[")
				a = temp[0][:l]
				a = a.replace(":","_")
				a = a.replace("?","=")
				a = a.replace("¿","!")
				a = a.replace("\n","")

				y = min([int(z) for z in list(re.findall("[1920][1920][0-9][0-9]", temp[0]))])
				a = str(y) + " - " + name[6:-4] + " - " + str(int(x/2)).rjust(2,'0') + " - " + a+ ".txt"

				print(a)

				d[a] = {"header":temp[0], "body":temp[1], "year": y}

				output_folder = output_folder + str(y) +"/"
				os.makedirs(output_folder, exist_ok=True)

				if save is None:
					if save or len(input("Are you sure of what you are about to do?\n\n this will require for you to fix by hand every file after.\n\nType something if ok.\nBTW, the code is probably commented\n\n"))>0:
						save = True
					else:
						save = False
				if save:
					# with open(	
					# 			output_folder + 
					# 			a,
					# 			'w', 
					# 			encoding='utf-8') as f:
					# 	f.write(temp[1])
					pass
			else:
				temp[0] = s[x]
				if temp[0] not in h:
					raise Exception("Wrong!! no matching heading")

if save:
	# with open("headings.json", 'w', encoding='utf-8') as f :
	# 	json.dump(d, f)
	pass