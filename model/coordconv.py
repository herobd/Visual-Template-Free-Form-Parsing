"""
    Copyright 2019 Brian Davis
    Visual-Template-free-Form-Parsting is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Visual-Template-free-Form-Parsting is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Visual-Template-free-Form-Parsting.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch
from torch import nn
import numpy as np

class CoordConv(nn.Module):
    def  __init__(self,in_ch,out_ch,kernel_size=3,padding=1,dilation=1,groups=1,features='wave'):
        super(CoordConv, self).__init__()
        self.features=features
        if 'wave' in features:
            if 'Big' in features:
                self.numChX=10
                self.numChY=7
                self.minCycle=8
                self.maxCycleX=2000
                self.maxCycleY=1400
            elif 'Med' in features:
                self.numChX=10
                self.numChY=7
                self.minCycle=4
                self.maxCycleX=1000
                self.maxCycleY=700
            elif 'Small' in features:
                self.numChX=10
                self.numChY=7
                self.minCycle=2
                self.maxCycleX=500
                self.maxCycleY=350
            else:
                self.numChX=5
                self.numChY=4
                self.minCycle=16
                self.maxCycleX=1000
                self.maxCycleY=700

            self.cycleStepX = (self.maxCycleX-self.minCycle)/((self.numChX-1)**2)
            self.cycleStepY = (self.maxCycleY-self.minCycle)/((self.numChY-1)**2)
            self.numExtra=self.numChX+self.numChY

        self.conv = nn.Conv2d(in_ch+self.numExtra,out_ch, kernel_size=kernel_size, padding=padding,dilation=dilation,groups=groups)

    def forward(self,input):
        batch_size = input.size(0)
        dimY=input.size(2)
        dimX=input.size(3)
        if 'wave' in self.features:
            if self.training:
                xOffset = np.random.randint(0,self.maxCycleX)
                yOffset = np.random.randint(0,self.maxCycleY)
            else:
                xOffset=0
                yOffset=0

            extraX = torch.FloatTensor(self.numChX,dimX)
            x_range = torch.arange(dimX, dtype=torch.float64) + xOffset
            for i in range(self.numChX):
                cycle = self.minCycle + self.cycleStepX*(i**2)
                extraX[i] = torch.sin(x_range*np.pi*2/cycle)
            extraX = extraX[:,None,:].expand(self.numChX,dimY,dimX)

            extraY = torch.FloatTensor(self.numChY,dimY)
            y_range = torch.arange(dimY, dtype=torch.float64) + yOffset
            for i in range(self.numChY):
                cycle = self.minCycle + self.cycleStepY*(i**2)
                extraY[i] = torch.sin(y_range*np.pi*2/cycle)
            extraY = extraY[:,:,None].expand(self.numChY,dimY,dimX)
            extra = torch.cat((extraY,extraX),dim=0)


        extra = extra[None,...].repeat(batch_size,1,1,1).to(input.device)
        data = torch.cat((input,extra),dim=1)

        return self.conv(data)
