#=**
* Custom Matrix Transpose Function
* Qijia (Michael) Jin
* @version 0.0.1
*
* Copyright (C) 2016  Qijia (Michael) Jin
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*=#

function transposeNC(matrix)#return transpose of matrix
	#matrix' gives the conjugate transpose, which we don't want
	if matrix == []
		return []
	else
		local tmp = vec(matrix[1,:])
		if size(matrix,1) > 1
			for i in 2:size(matrix,1)
				tmp = hcat(tmp,vec(matrix[i,:]))
			end
		end
		return tmp
	end
end