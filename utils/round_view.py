from typing import Tuple
import numpy as np
from enum import Enum

def generate_mask(mask, r):
    assert mask.ndim == 2
    assert mask.shape[0] == mask.shape[1]
    assert mask.shape[0] % 2 == 1
    
    mask = mask.copy()
    
    l = mask.shape[0]
    c = (l + 1) // 2
    index_list = []
    
    for u in range(l):
        for v in range(l):
            if (u + 1 - c) ** 2 + (v + 1 - c) ** 2 <= r ** 2:
                mask[u, v] = 1
                index_list.append(u * l + v)
    return mask, index_list

class D(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

class RoundView:
    mask:np.ndarray
    max_row = -1
    size = -1
    total_view = -1
    
    
    def __init__(self, radius:int, total_view:int=13) -> None:
        self.mask = np.zeros((total_view, total_view), dtype=np.short)
        self.mask, self.lf_2_rv_index_list = generate_mask(self.mask, radius)
        self.width_per_row = []
        for row in self.mask:
            s = row.sum()
            if s != 0:
                self.width_per_row.append(s)
        
        self.max_row = max(self.width_per_row)
        self.size = self.mask.sum()
        self.total_view = total_view
        
        self.precache()
    
    def precache(self):
        # normal xs xt
        self.o_2_xs_index_list = np.zeros(self.size, dtype=int)
        self.o_2_xt_index_list = np.zeros(self.size, dtype=int)
        self.xs_2_o_index_list = np.zeros(self.size, dtype=int)
        self.xs_2_xt_index_list = np.zeros(self.size, dtype=int)
        self.xt_2_xs_index_list = np.zeros(self.size, dtype=int)
        o_i = 0
        for u, width in enumerate(self.width_per_row):
            for v in range(width):
                xs_i = self._uv_2_xs_i(u, v)
                xt_i = self._uv_2_xt_i(u, v)
                self.o_2_xs_index_list[xs_i] = o_i
                self.o_2_xt_index_list[xt_i] = o_i
                self.xs_2_o_index_list[o_i] = xs_i
                self.xs_2_xt_index_list[xt_i] = xs_i
                self.xt_2_xs_index_list[xs_i] = xt_i
                o_i += 1
        # clockwise xs xt
        self.o_2_cxs_index_list = np.array(self.generate_clockwise_map(clockwise=True), dtype=int)
        self.o_2_cxt_index_list = np.array(self.generate_clockwise_map(clockwise=False), dtype=int)
        self.cxs_2_o_index_list = np.zeros(self.size, dtype=int)
        self.cxs_2_cxt_index_list = np.zeros(self.size, dtype=int)
        self.cxt_2_cxs_index_list = np.zeros(self.size, dtype=int)
        for o_i in range(self.size):
            xs_i = np.where(self.o_2_cxs_index_list == o_i)[0][0]
            xt_i = np.where(self.o_2_cxt_index_list == o_i)[0][0]
            
            self.cxs_2_cxt_index_list[xt_i] = xs_i
            self.cxt_2_cxs_index_list[xs_i] = xt_i
            self.cxs_2_o_index_list[o_i] = xs_i
    
    def rv_stack_2_lf(self, stack:np.ndarray):
        assert stack.ndim == 3
        assert stack.shape[0] == self.size
        assert stack.shape[1] == stack.shape[2]
        
        res = np.zeros((self.total_view * self.total_view, *stack.shape[1:]), dtype=stack.dtype)
        res[self.lf_2_rv_index_list] = stack
        
        return res.reshape(self.total_view, self.total_view, *stack.shape[1:])


    def _uv_2_xs_i(self, u:int, v:int) -> int:
        xs_i = 0
        for y in range(u):
            xs_i += self.width_per_row[y]
    
        if u % 2 == 0:
            return xs_i + v
        else:
            return xs_i + self.width_per_row[u] - 1 - v

    def _uv_2_padded_uv(self, u:int, v:int) -> Tuple[int, int]:
        p = (self.max_row - self.width_per_row[u]) // 2
        return u, p + v

    def _padded_uv_2_uv(self, u:int, v:int) -> Tuple[int, int]:
        assert 0 <= u < self.max_row
        assert 0 <= v < self.max_row
        p = (self.max_row - self.width_per_row[u]) // 2
        assert 0 <= v - p < self.width_per_row[u]
        return u, v - p
    

    def _uv_2_xt_i(self, u:int, v:int) -> int:
        # xs coor to xt coor
        u, v = self._uv_2_padded_uv(u, v)
        u, v = v, u     
        u, v= self._padded_uv_2_uv(u, v)
    
        return self._uv_2_xs_i(u, v)
    
    def _lf_uv_2_o_i(self, u:int, v:int) -> int:
        row_pad = self.mask.shape[0] - len(self.width_per_row)
        assert row_pad % 2 == 0
        row_pad //= 2
        u -= row_pad
        assert u >= 0
        true_w = self.width_per_row[u]
        col_pad = self.mask.shape[1] - true_w
        assert col_pad % 2 == 0
        col_pad //= 2
        v -= col_pad
        assert v >= 0
        o_i = sum(self.width_per_row[0:u]) + v
        return o_i

    def generate_clockwise_map(self, clockwise=True):
        h, w = self.mask.shape
        assert h == w
        assert h % 2 == 1

        mid = (h - 1) // 2
        u, v = mid, mid # This is LF uv, not converted uv used in other places in this class
        m = np.ones_like(self.mask, dtype=np.int64)
        m *= - 1
        d_map = {
            D.LEFT: D.UP,
            D.UP: D.RIGHT,
            D.RIGHT: D.DOWN,
            D.DOWN: D.LEFT
        } if clockwise else \
        {
            D.RIGHT: D.UP,
            D.UP: D.LEFT,
            D.LEFT: D.DOWN,
            D.DOWN: D.RIGHT
        }

        def go(u, v, d):
            if d == D.UP:
                return u - 1, v
            elif d == D.DOWN:
                return u + 1, v
            elif d == D.LEFT:
                return u, v - 1
            elif d == D.RIGHT:
                return u, v + 1

        def can_go(mask, u, v, d):
            u, v = go(u, v, d)
            if u < 0 or v < 0 or u >= mask.shape[0] or v >= mask.shape[1]:
                return False
            return mask[u, v] == -1

        stepped = 0
        total = h * w
        d = D.LEFT if clockwise else D.RIGHT
        res = []
        while stepped < total:
            m[u, v] = stepped
            if self.mask[u, v]:
                res.append(self._lf_uv_2_o_i(u, v))
        
            stepped += 1
        
            u, v = go(u,v,d)
            nwd = d_map[d]
            if can_go(m, u, v, nwd):
                d = nwd
        return res

def main():
    rv = RoundView(5)
    
    lf = np.arange(13 * 13)
    print(rv.mask)
    v = lf[rv.lf_2_rv_index_list]
    print(v)
    v = v[rv.o_2_cxs_index_list]
    v = v[rv.cxs_2_cxt_index_list]
    v = v[rv.cxt_2_cxs_index_list]
    v = v[rv.cxs_2_o_index_list]
    
    r = np.zeros(13 * 13)
    r[rv.lf_2_rv_index_list] = v
    print(r.reshape(13, 13).astype(int))
    print()
    print(np.arange(169).reshape(13, 13))
    
    # a = np.arange(81)
    # xs = a[rv.o_2_cxs_index_list]
    # xt = a[rv.o_2_cxt_index_list]
    # xt_f_xs = xs[rv.cxs_2_cxt_index_list]
    # xs_f_xt = xt[rv.cxt_2_cxs_index_list]
    # ap = xs[rv.cxs_2_o_index_list]
    
    # print(a.reshape(9, 9))
    # print()
    # print(xs.reshape(9, 9))
    # print(xs_f_xt.reshape(9, 9))
    # print()
    # print(xt.reshape(9, 9))
    # print(xt_f_xs.reshape(9, 9))
    # print()
    # print(ap.reshape(9, 9))
    
    

if __name__ == "__main__":
    main()