function FunctionDefinition_0 ( uint256 _value ) returns ( bool success ) { Identifier_0 [ msg . sender ] -= _value ; balances [ msg . sender ] += _value ; Identifier_1 ( msg . sender , _value ) ; return true ; }