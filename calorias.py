import numpy as np
import pandas as pd
from pandas import read_csv
from IPython.display import display
from time import time
import ipywidgets
import matplotlib.pyplot as plt
import ipywidgets

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def restart():
    file = open('historia', 'w')
    #file.write('')
    file.write('date,name,g,cal,carb,prota,protv,fib,sf,muf,puf,vA,vB1,vB2,vB3,vB5,vB6,vB9folate,vB12,vC,vD,vE,vK,Mn,Mg,P,K,Ca,Se,Cu,Fe,Zn\n\n')
    file.close()

day2000 = np.array([2000,275,60,30,70]) #cal carb prot fib grasa
plan = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
actualplan= np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

####
class Comida:
    def __init__(self,name,cal,carb,prota,protv,fib,sf,muf,puf,vA,vB1,vB2,vB3,vB5,vB6,vB9folate,vB12,vC,vD,vE,vK,Mn,Mg,P,K,Ca,Se,Cu,Fe,Zn):
        self.nombre = name
        self.valor = np.array([cal,carb,prota,protv,fib,sf,muf,puf,vA,vB1,vB2,vB3,vB5,vB6,vB9folate,vB12,vC,vD,vE,vK,Mn,Mg,P,K,Ca,Se,Cu,Fe,Zn])
    
    def eatg(self,g,date):
        file = open('historia', 'a')
        file.write(f"{date},{self.nombre},{g},{np.array2string((self.valor/100.)*g,separator=', ')[1:-1]}\n")
        file.close()

    def onlyvalor(self,g):
      return ((self.valor/100.)*g)


    def eatn(self,n,divisor,date):
        file = open('historia', 'a')
        file.write(f",{date},{self.nombre},{n},{np.array2string((self.valor/divisor)*n,separator=', ')[1:-1]}\n")
        file.close()

    #def plan(self,g,cal):  
        # factorplan=cal/2000
        # totalday= day2000 * factorplan
        # total = totalday/2

        # labels = ['cal','carb','prota','protv','fib','sf','muf','puf','vA','vB1','vB2','vB3','vB5','vB6','vB9folate','vB12','vC','vD','vE','vK','Mn','Mg','P','K','Ca','Se','Cu','Fe','Zn']
        # figplan = plt.figure()
  
        # axplan = figplan.add_axes([0,0,1,1])
        # plt.axhline(100)
        # axplan.bar(labels,(self.valor*g) /  (np.append(total, [[4, 5, 6], [7, 8, 9]]))   ))
        # plt.show()



    def addtoplan(self,g):
      valores = ['cal','carb','prota','protv','fib','sf','muf','puf','vA','vB1','vB2','vB3','vB5','vB6','vB9folate','vB12','vC','vD','vE','vK','Mn','Mg','P','K','Ca','Se','Cu','Fe','Zn']
      figv = plt.figure()
      axv = figv.add_axes([0,0,1,1])
      plt.axhline(100)
      axv.bar(valores,(self.valor/100)*g)
#####     


def plan(dias=7,arroz_gallo_oro1=0,alcachofa1=0,almendra1=0,ajo1=0,batata1=0,berenjena1=0,brocoli1=0,carne_vaca1=0,caju1=0,cebolla1=0,col_brusela1=0,coliflor1=0,esparrago1=0,espinaca1=0,huevo1=0,leche_lasuipachence_ml1=0,lechuga_francesa1=0,lechuga_manteca1=0,maiz1=0,manzana1=0,morron1=0,nuez1=0,palta1=0,papa1=0,pepino1=0,puerro1=0,pollo_pechuga1=0,rabano1=0,remolacha1=0,rucula1=0,tomate1=0,zanahoria1=0,zapallo_anco1=0):
  valores = ['cal','carb','prota','protv','fib','sf','muf','puf','vA','vB1','vB2','vB3','vB5','vB6','vB9folate','vB12','vC','vD','vE','vK','Mn','Mg','P','K','Ca','Se','Cu','Fe','Zn']
  #comida2 = arroz_gallo_oro.onlyvalor(arroz_gallo_oro),alcachofa.onlyvalor(alcachofa),almendra.onlyvalor(almendra),ajo.onlyvalor(ajo),batata.onlyvalor(batata),berenjena.onlyvalor(berenjena),brocoli.onlyvalor(brocoli),carne_vaca.onlyvalor(carne_vaca),caju.onlyvalor(caju),cebolla.onlyvalor(cebolla),col_brusela.onlyvalor(col_brusela),coliflor.onlyvalor(coliflor),esparrago.onlyvalor(esparrago),espinaca.onlyvalor(espinaca),huevo.onlyvalor(huevo),leche_lasuipachence_ml.onlyvalor(leche_lasuipachence_ml),lechuga_francesa.onlyvalor(lechuga_francesa),lechuga_manteca.onlyvalor(lechuga_manteca),maiz.onlyvalor(maiz),manzana.onlyvalor(manzana),morron.onlyvalor(morron),nuez.onlyvalor(nuez),palta.onlyvalor(palta),papa.onlyvalor(papa),pepino.onlyvalor(pepino),puerro.onlyvalor(puerro),pollo_pechuga.onlyvalor(pollo_pechuga),rabano.onlyvalor(rabano),remolacha.onlyvalor(remolacha),rucula.onlyvalor(rucula),tomate.onlyvalor(tomate),zanahoria.onlyvalor(zanahoria),zapallo_anco.onlyvalor(zapallo_anco))
  #comida3 = arroz_gallo_oro.onlyvalor(arroz_gallo_oro)
  #print(arroz_gallo_oro.onlyvalor(arroz_gallo_oro1))
#day2000 = np.array([2000,275,60,30,70]) #cal carb prot fib grasa
  comida = arroz_gallo_oro.onlyvalor(arroz_gallo_oro1)+alcachofa.onlyvalor(alcachofa1)+almendra.onlyvalor(almendra1)+ajo.onlyvalor(ajo1)+batata.onlyvalor(batata1)+berenjena.onlyvalor(berenjena1)+brocoli.onlyvalor(brocoli1)+carne_vaca.onlyvalor(carne_vaca1)+caju.onlyvalor(caju1)+cebolla.onlyvalor(cebolla1)+col_brusela.onlyvalor(col_brusela1)+coliflor.onlyvalor(coliflor1)+esparrago.onlyvalor(esparrago1)+espinaca.onlyvalor(espinaca1)+huevo.onlyvalor(huevo1)+leche_lasuipachence_ml.onlyvalor(leche_lasuipachence_ml1)+lechuga_francesa.onlyvalor(lechuga_francesa1)+lechuga_manteca.onlyvalor(lechuga_manteca1)+maiz.onlyvalor(maiz1)+manzana.onlyvalor(manzana1)+morron.onlyvalor(morron1)+nuez.onlyvalor(nuez1)+palta.onlyvalor(palta1)+papa.onlyvalor(papa1)+pepino.onlyvalor(pepino1)+puerro.onlyvalor(puerro1)+pollo_pechuga.onlyvalor(pollo_pechuga1)+rabano.onlyvalor(rabano1)+remolacha.onlyvalor(remolacha1)+rucula.onlyvalor(rucula1)+tomate.onlyvalor(tomate1)+zanahoria.onlyvalor(zanahoria1)+zapallo_anco.onlyvalor(zapallo_anco1)  
  objetivo = np.array([2000,275,60,60,30,70,70,70,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]) * dias
  #print(comida)
  #plt.plot(comida2)
  #axv.bar(valores,np.add(arroz_gallo_oro.onlyvalor(arroz_gallo_oro),alcachofa.onlyvalor(alcachofa),almendra.onlyvalor(almendra),ajo.onlyvalor(ajo),batata.onlyvalor(batata),berenjena.onlyvalor(berenjena),brocoli.onlyvalor(brocoli),carne_vaca.onlyvalor(carne_vaca),caju.onlyvalor(caju),cebolla.onlyvalor(cebolla),col_brusela.onlyvalor(col_brusela),coliflor.onlyvalor(coliflor),esparrago.onlyvalor(esparrago),espinaca.onlyvalor(espinaca),huevo.onlyvalor(huevo),leche_lasuipachence_ml.onlyvalor(leche_lasuipachence_ml),lechuga_francesa.onlyvalor(lechuga_francesa),lechuga_manteca.onlyvalor(lechuga_manteca),maiz.onlyvalor(maiz),manzana.onlyvalor(manzana),morron.onlyvalor(morron),nuez.onlyvalor(nuez),palta.onlyvalor(palta),papa.onlyvalor(papa),pepino.onlyvalor(pepino),puerro.onlyvalor(puerro),pollo_pechuga.onlyvalor(pollo_pechuga),rabano.onlyvalor(rabano),remolacha.onlyvalor(remolacha),rucula.onlyvalor(rucula),tomate.onlyvalor(tomate),zanahoria.onlyvalor(zanahoria),zapallo_anco.onlyvalor(zapallo_anco)))
  #return comida2
  plt.figure(figsize=(15, 5), dpi=80)
  plt.bar(valores,comida,alpha=.6)
  plt.bar(valores,objetivo,alpha=.6)
  plt.show()
  

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


a=0
b=0
#comida = Comida("comida",cal,carb,prota,protv,fib,sf,muf,puf,vA,vB1,vB2,vB3,vB5,vB6,vB9folate,vB12,vC,vD,vE,vK,Mn,Mg,P,K,Ca,Se,Cu,Fe,Zn)
arroz_gallo_oro = Comida("arroz_gallo_oro",332,76,a,7,1,0,0,0,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b)
alcachofa = Comida("alcachofa",55,12,a,a,9,a,a,a,b,b,b,b,b,b,22,b,12,b,b,18,11,10,7,8,b,b,b,b,b)
almendra = Comida("almendra",575,21,0,21,12,4,31,12,b,b,60,b,b,b,b,b,b,b,131,b,114,67,48,20,b,b,50,21,21)
ajo = Comida("ajo",150,33,a,a,2,a,a,a,b,b,b,b,b,62,b,b,52,b,b,b,84,b,15,b,18,20,15,b,b)
batata = Comida("batata",75,18,a,a,3,a,a,a,315,b,b,b,6,8,b,b,21,b,5,b,13,b,b,7,b,b,b,b,b)

berenjena = Comida("berenjena",35,9,a,a,3,a,a,a,b,5,b,b,b,4,3,b,3,b,b,4,6,b,b,4,b,b,b,b,b)

brocoli = Comida("brocoli",35,7,a,a,3,a,a,a,31,b,b,b,b,10,27,b,108,b,b,176,10,b,b,8,b,b,b,b,b)
carne_vaca = Comida("carne_vaca",288,a,26,0,a,1,a,a,b,b,b,35,b,28,b,170,b,b,b,b,b,5,9,9,b,41,b,9,43)

caju = Comida("caju",575,31,0,16,3,9,27,8,b,b,b,b,b,b,b,b,b,b,b,43,41,65,49,16,b,17,111,33,37)
cebolla = Comida("cebolla",45,10,a,a,1,a,a,a,b,b,b,b,b,6,b,b,9,b,b,b,8,b,4,5,b,b,3,b,b)

col_brusela = Comida("col_brusela",35,7,a,a,3,a,a,a,15,b,b,b,b,9,15,b,103,b,b,175,11,b,9,b,b,b,b,b,b)
coliflor = Comida("coliflor",25,4,a,a,2,a,a,a,b,b,b,b,5,9,11,b,74,b,b,17,7,b,b,4,b,b,b,b,b)



esparrago = Comida("esparrago",20,4,a,a,2,a,a,a,20,11,b,b,b,b,37,b,13,b,b,64,b,b,b,b,b,9,8,b,b)
espinaca = Comida("espinaca",25,4,a,a,2,a,a,a,210,b,b,b,b,b,37,b,16,b,b,617,47,22,b,b,b,b,b,20,b)
huevo = Comida("huevo",143,0.72,12,0,0,4,5,2,15,b,30,b,16,8,13,23,b,9,b,b,b,b,21,b,b,49,b,11,b)
leche_lasuipachence_ml = Comida("leche_lasuipachence_ml",40,4.7,3,0,b,0.6,b,b,20,b,b,b,b,b,b,b,40,40,b,b,b,b,b,b,b,b,b,b,b)
lechuga_manteca = Comida("lechuga_manteca",15,3,a,a,1,a,a,a,10,3,b,b,b,b,7,b,5,b,b,30,6,b,b,4,b,b,b,b,b)
lechuga_francesa = Comida("lechuga_francesa",15,3,a,a,2,b,b,b,174,b,b,b,b,b,34,b,40,b,b,b,8,b,b,7,b,b,b,5,b)


maiz = Comida("maiz",110,25,a,a,3,a,a,a,b,14,b,8,9,b,11,b,10,b,b,b,8,b,7,b,b,b,b,b,b)
manzana = Comida("manzana",50,10,a,a,3,a,a,a,1,b,2,b,b,2,b,b,8,b,b,3,2,b,b,3,b,b,b,b,b)
morron = Comida("morron",30,7,a,a,1,a,a,a,63,b,b,b,b,15,11,b,213,b,8,6,b,b,b,6,b,b,b,b,b)
nuez = Comida("nuez",655,14,0,15,7,6,9,47,b,23,b,b,b,27,25,b,b,b,b,b,171,40,35,13,b,b,79,16,21)
palta = Comida("palta",160,a,a,a,7,a,a,a,b,b,b,b,14,13,20,b,17,b,10,26,b,b,b,14,b,b,b,b,b)
papa = Comida("papa",85,20,a,a,2,a,a,a,b,7,b,7,b,15,b,b,22,b,b,b,7,b,b,11,b,b,9,b,b)
pepino = Comida("pepino",15,4,a,a,1,a,a,a,b,b,b,b,3,b,b,b,5,b,b,21,4,3,b,4,b,b,2,b,b)
puerro = Comida("puerro",30,8,a,a,1,a,a,a,16,b,b,b,b,6,6,b,7,b,b,32,12,b,b,b,b,b,b,6,b)
pollo_pechuga = Comida("pollo_pechuga",150,0,25,0,0,1,1,1,b,b,b,69,b,30,b,6,b,b,b,b,b,7,23,7,b,39,b,6,7)
rabano = Comida("rabano",15,4,a,a,2,a,a,a,b,b,b,b,b,4,6,b,25,b,b,b,3,b,b,7,b,b,3,b,2)

remolacha = Comida("remolacha",45,10,a,a,2,a,a,a,b,b,b,b,b,b,20,b,6,b,b,b,26,6,b,9,b,b,4,4,b)

rucula = Comida("rucula",25,4,a,a,2,a,a,a,47,b,b,b,b,b,24,b,25,b,b,136,16,12,b,b,16,b,b,b,b)
tomate = Comida("tomate",20,a,a,a,1,a,a,a,17,a,a,a,a,4,4,b,21,b,b,10,6,b,b,b,b,b,b,b,b)
zanahoria = Comida("zanahoria",35,8,a,a,3,a,a,a,341,b,b,b,b,8,b,b,6,b,5,17,8,b,b,7,b,b,b,b,b)
zapallo_anco = Comida("zapallo_anco",40,a,a,a,3,a,a,a,223,b,b,b,b,6,b,b,25,b,6,b,9,7,b,8,b,b,b,b,b)


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


def macro(cal,dia):
  factor=cal/2000
   #caloria, carbohidrato, proteina, fibra, grasa
  realday= day2000*factor
  df = read_csv('historia', sep=',', skiprows=0, decimal='.',index_col=False)
  hasta_dia = df[df['date'] >= dia]
  n=1
  for x in range(1, hasta_dia.shape[0]):
    if hasta_dia.iloc[x-1,0] != hasta_dia.iloc[x,0]:
      n = n +1
  plt.figure(figsize=(6,4))
  plt.bar("cal", hasta_dia['cal'].sum(0) / (realday[0] *n))
  plt.bar("carb", hasta_dia['carb'].sum(0) / (realday[1] *n))
  plt.bar("prot",(hasta_dia['prota'].sum(0) + hasta_dia['protv'].sum(0) )/ (realday[2]*n))
  plt.bar("grasa", (hasta_dia['sf'].sum(0) + hasta_dia['muf'].sum(0) + hasta_dia['puf'].sum(0)) / (realday[4]*n))
  plt.bar("fibra",hasta_dia['fib'].sum(0)/(realday[3]*n))
  plt.ylim(0,1.1)
  plt.axhline(1)
  plt.show()

def mineral(cal,dia):
  factor=cal/2000
  
  realday= day2000*factor
  df = read_csv('historia', sep=',', skiprows=0, decimal='.',index_col=False)
  hasta_dia = df[df['date'] >= dia]
  n=1
  for x in range(1, hasta_dia.shape[0]):
    if hasta_dia.iloc[x-1,0] != hasta_dia.iloc[x,0]:
      n = n +1
  mineral_hasta_dia = np.array([hasta_dia['Mn'].sum(0),hasta_dia['Mg'].sum(0),hasta_dia['P'].sum(0),hasta_dia['K'].sum(0),hasta_dia['Ca'].sum(0),hasta_dia['Se'].sum(0),hasta_dia['Cu'].sum(0),hasta_dia['Fe'].sum(0),hasta_dia['Zn'].sum(0)])
  minerales = ["Mn","Mg","P","K","Ca","Se","Cu","Fe","Zn"]
  figm = plt.figure()
  axm = figm.add_axes([0,0,1,1])
  plt.axhline(100)
  axm.bar(minerales,mineral_hasta_dia/n)
  plt.show()

def vitamina(cal,dia):
  factor=cal/2000
  
  realday= day2000*factor
  df = read_csv('historia', sep=',', skiprows=0, decimal='.',index_col=False)
  hasta_dia = df[df['date'] >= dia]
  n=1
  for x in range(1, hasta_dia.shape[0]):    
    if hasta_dia.iloc[x-1,0] != hasta_dia.iloc[x,0]:
      n = n +1
  vit_hasta_dia = np.array([hasta_dia['vA'].sum(0),hasta_dia['vB1'].sum(0),hasta_dia['vB2'].sum(0),hasta_dia['vB3'].sum(0),hasta_dia['vB5'].sum(0),hasta_dia['vB6'].sum(0),hasta_dia['vB9folate'].sum(0),hasta_dia['vB12'].sum(0),hasta_dia['vC'].sum(0),hasta_dia['vD'].sum(0),hasta_dia['vE'].sum(0),hasta_dia['vK'].sum(0)])
  vitaminas = ["A",'B1','B2','B3','B5','B6','B9','B12','C','D','E','K']
  figv = plt.figure()
  axv = figv.add_axes([0,0,1,1])
  plt.axhline(100)
  axv.bar(vitaminas,vit_hasta_dia/n)
  plt.show()

def full(cal,dia):
  macro(cal,dia)
  mineral(cal,dia)
  vitamina(cal,dia)


restart()

################ Historia

remolacha.eatg(50,230127.13)
brocoli.eatg(100,230114.13)
huevo.eatg(240,230114.13)
puerro.eatg(1000,230114.13)
manzana.eatg(200,230114.13)
batata.eatg(200,230114.19)
rabano.eatg(500,230114.19)

caju.eatg(100,230120.13)
almendra.eatg(240,230120.13)
zapallo_anco.eatg(1000,230120.13)
papa.eatg(200,230120.19)
pollo_pechuga.eatg(200,230120.19)
rucula.eatg(500,230120.19)

pepino.eatg(100,230122.13)
leche_lasuipachence_ml.eatg(240,230122.13)
papa.eatg(1000,230122.13)
ajo.eatg(200,230122.13)
arroz_gallo_oro.eatg(200,230122.13)
rabano.eatg(500,230122.13)

tomate.eatg(100,230121.19)
zanahoria.eatg(240,230121.19)
carne_vaca.eatg(1000,230121.19)
papa.eatg(200,230121.19)
lechuga_francesa.eatg(200,230121.19)
remolacha.eatg(500,230121.19)

full(2000, 241214)

Comida.eatg("remolacha", 70, 250224)