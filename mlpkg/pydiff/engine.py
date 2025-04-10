class Value():

    def __init__(self , data : float , _children : tuple['Value' , ...] = () , _op : str = None , label : str = ""):
        self.data = data
        self._op = _op
        self._children = set(_children)
        self.label = label
        self.grad = 0

        self._backward = lambda : None
        self._forward = lambda : None
    
    def __repr__(self):
        return f"Value(data={self.data})"

    @staticmethod
    def _coercion(other):
        if isinstance(other , Value):
            return other
        elif isinstance(other , (int, float)):
            return Value(data=float(other))
        else:
            return None


    def __add__(self , other):
        other = self._coercion(other)

        if (other is None ) : return NotImplemented

        out = Value(self.data + other.data , _children=(self , other) , _op = '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        def _forward():
            out.data = self.data + other.data
        
        out._forward = _forward
        out._backward = _backward

        return out

    def __radd__(self , other):

        return other + self

    def __mul__(self , other):
        other = self._coercion(other)

        if (other is None ) : return NotImplemented

        out = Value(self.data * other.data , _children=(self , other) , _op = '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        def _forward():
            out.data = self.data * other.data
        
        out._forward  = _forward
        out._backward = _backward

        return out

    def __rmul__(self , other):

        return other * self

    def __pow__(self , k):
        if not isinstance(k , (int , float)):
            raise TypeError(f"Unsupported operand {k} of Type {type(k)}; Expected float , int.")

        out = Value(data= self.data ** k , _children=(self,) , _op = f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k-1)) * out.grad 
        
        def _forward():
            out.data = self.data ** k

        out._forward = _forward
        out._backward = _backward
        
        return out


    # using predefined function
    def __neg__(self):
        return self * -1
    
    def __sub__(self , other):
        return self + (-other)

    def __rsub__(self , other):
        return other + (-self)

    def __truediv__(self , other):
        return self * (other**-1)

    def __rtruediv__(self , other):
        return other * (self**-1)

    #back propagation
    @staticmethod
    def _topological_sort(root : 'Value'):

        visited = set()
        result = []

        def _topo(v : 'Value'):

            if v not in visited:
                visited.add(v)
                for child in v._children:
                    _topo(child)
                result.append(v)
        _topo(root)

        return result
    
    def backward(self):
        topo_order = self._topological_sort(self)

        self.grad = 1.0

        for node in reversed(topo_order):
            node._backward()
    
    def zero_grads(self):

        topo_order = self._topological_sort(self)

        for node in topo_order:
            node.grad = 0.0
    
    def forward(self):

        topo_order = self._topological_sort(self)

        for node in topo_order:
            node._forward()