def LZWEncode(codes):
	i,P,numbers=1,"",""
	d={}

	for code in codes:
		if d.get(code)==None:
			d[code]=i
			i=i+1
	for C in codes:
		if d.get(P+C)==None:
			numbers=numbers+str(d.get(P))
			d[P+C]=i
			i=i+1
			P=C
		else:
			P=P+C
	print("源码为:",codes)
	print("编码为:",numbers)
	print("字典为:",d)
	return d
def LZWDecode(dict,numbers):
	p,tag="",0
	for number in numbers:
		for d in dict:
			if dict[d]== number:
				p=p+d
				tag=1
				break
		if tag==0:
			return "error"
	return p

i=0
codes=["ABBABAADDABCCA","AZCASWQDFAVASDASD","ASASFQWDASCZXC","ZCASCZCZXCS"]
for code in codes:
	d=LZWEncode(codes)
	i+=1
numbers=[1,2,3,6,9,1,12,4,5]
print("编码为:",numbers)
print("解码为:",LZWDecode(d,numbers))
