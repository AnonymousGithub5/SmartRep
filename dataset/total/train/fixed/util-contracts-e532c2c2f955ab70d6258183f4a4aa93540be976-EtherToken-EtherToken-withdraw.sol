function withdraw ( uint value ) public { balances [ msg . sender ] = balances [ msg . sender ] . sub ( value ) ; Identifier_0 = Identifier_1 . sub ( value ) ; msg . sender . transfer ( value ) ; emit Identifier_2 ( msg . sender , value ) ; }