import jsonlines

ment2ent = {}
with jsonlines.open("data/ment2ent.txt") as fin:
	for item in fin:
		ment = item[0]
		ents = [ment if e is None else e for e in item[1]][:50]
		if ment.strip() and ents: ment2ent[ment] = ents
print(len(ment2ent), "ment2ents")