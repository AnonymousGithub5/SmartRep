function approve ( address spender , uint value ) public returns ( bool ) { Identifier_0 [ msg . sender ] [ spender ] = value ; emit Identifier_1 ( msg . sender , spender , value ) ; return true ; }