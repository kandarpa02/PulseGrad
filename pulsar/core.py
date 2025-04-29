import numpy as np

class pulse:
    def __init__(self, data, _children=[], op='', compute_grad = False, size = 1):
        self.data = data
        self.size = (len(self.data), len(self.data[0])) if isinstance(self.data, (list, np.ndarray)) else size
        self.gradient = np.zeros_like(self.data, dtype=np.float32) if isinstance(self.data, (list, np.ndarray)) else 0

        self._back = lambda: None
        self.stored = list(_children)
        self.op = op
        self.compute_grad = compute_grad
        
    def __repr__(self):
        if isinstance(self.data, (list, np.ndarray)):
        
            matrix = self.data
            head, tail = 3, 2
            total_rows = len(matrix)
            prefix = "pulse("
            indent = " " * (len(prefix) + 1)
        
            def fmt_row(row):
                vals = ", ".join(f"{v:7.4f}" for v in row)
                return f"[ {vals} ]"
        
            if total_rows <= head + tail:
                rows = [fmt_row(r) for r in matrix]
            else:
                rows = [fmt_row(r) for r in matrix[:head]] + ["..."] + [fmt_row(r) for r in matrix[-tail:]]
        
            if len(rows) == 1:
                block = "[" + rows[0] + "]"
            else:
                lines = []
               
                lines.append(prefix + "[" + rows[0] + ",")
                for row in rows[1:-1]:
                    lines.append(indent + (row + "," if row != "..." else "...,")) 
                
                lines.append(indent + rows[-1] + "]")
                block = "\n".join(lines)

            com_grad = f", compute_grad: {'enabled' if self.compute_grad == True else 'disabled'})"
        
            return block + com_grad + ")"

        return f"pulse({self.data}, compute_grad: {'enabled' if self.compute_grad == True else 'disabled'})"


    def __add__(self, other):
        out = pulse(self.data + other.data, (self, other), '+', compute_grad= True if self.compute_grad == True else False)

        def _back():
            if self.compute_grad == True:
                self.gradient += 1.0 * out.gradient
                other.gradient += 1.0 * out.gradient
            else:
                raise ValueError("Please activate your Sharingan! you did not set 'compute_grad = True' before backprop")
        out._back = _back
        return out
 
    def __mul__(self, other):
        out = pulse(self.data * other.data, (self, other), '*', compute_grad= True if self.compute_grad == True else False)
        
        def _back():
            if self.compute_grad == True:
                self.gradient += out.gradient * other.data
                other.gradient += out.gradient * self.data
            else:
                raise ValueError("Please activate your Sharingan! you did not set 'compute_grad = True' before backprop")
        out._back = _back
        return out
        
    def __matmul__(self, other):
        if not isinstance(self, np.ndarray):
            self.data = np.array(self.data, dtype= np.float32)
            other.data = np.array(other.data, dtype= np.float32)
            
        m, k = self.data.shape 
        k_, n = other.data.shape

        if k!=k_:
            raise ValueError(f"Uhh Vibes didn't match!! :( please check the dimensions ( ,{k}) != ({k_}, )")
        
        result = np.dot(self.data, other.data)
        out = pulse(result, (self, other), '@', compute_grad=True if self.compute_grad else False, size=(m, n))
        
        def _back():
            if  self.compute_grad == True:
                self.gradient += np.dot(out.gradient, other.data.T)
                other.gradient += np.dot(self.data.T, out.gradient) 
            else:
                raise ValueError("Please activate your Sharingan! you did not set 'compute_grad = True' before backprop")
        out._back = _back
        return out

    def backprop(self):
        self.gradient = np.ones_like(self.data) if isinstance(self.data, np.ndarray) else 1
        topo = []
        visited = set()
        def build_topo(v):
            if v not in topo:
                visited.add(v)
                for child in v.stored:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node._back()