import csv
import matplotlib.pyplot as plt

x_index = [(i + 1) * 0.2 for i in range(10)]

x_index = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
entropy_results = [41.33570906, 57.54100877, 66.19943348, 72.55862573, 77.10630482, 80.91756213, 83.76920687, 87.58426535, 92.19864766]
deepgini_results = [38.71377924, 55.00031067, 64.10891813, 70.68645833, 75.61725146, 79.72836257, 82.91111111, 86.98364401, 91.9558845]
mcp_results = [29.9878655, 42.64340278, 51.31114766, 58.11884137, 63.62721126, 68.95003655, 72.95082237, 78.27574927, 84.02176535]
ces_results = [15.64471857, 25.6754386, 35.36652047, 44.70917398, 53.68289474, 62.81138523, 71.01779971, 79.96869518, 89.28263889]
dsa_results = [11.0464364, 20.7875, 31.45718202, 42.71880482, 52.53042763, 62.05926535, 69.56107456, 79.63031798, 89.58377193]
random_results = [10.80953947, 20.80188231, 30.42269737, 40.59546784, 50.58070175, 60.12107091, 69.05281433, 79.20062135, 89.35880848]
baseline_results = [10, 20, 30, 40, 50, 60, 70, 80, 90]



# with open('results/yelp_gru_com_win0.02.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         third.append(float(row[0]))

plt.plot(x_index, entropy_results, marker='o', label="Entropy")
plt.plot(x_index, deepgini_results, marker='o', label="DeepGini")
plt.plot(x_index, mcp_results, marker='o', label="MCP")
plt.plot(x_index, ces_results, marker='o', label="CES")
plt.plot(x_index, dsa_results, marker='o', label="DSA")
plt.plot(x_index, random_results,  marker='o', label="Random")

plt.plot(x_index, baseline_results, '--', marker='o', label="Baseline", color='black')
# plt.plot(x_index, third_fi, label="combine")

# plt.plot(x_index, first, label="without diversity")
# plt.plot(x_index, second, label="with diversity")
# plt.plot(x_index, third, label="combine")
axis_font = {'size': '14'}
plt.xlabel('% OOD data in the candidate set', **axis_font)
plt.ylabel('% OOD data in the selected set', **axis_font)
plt.grid()
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend()

plt.savefig('../imgs/RQ3.pdf')
plt.show()


