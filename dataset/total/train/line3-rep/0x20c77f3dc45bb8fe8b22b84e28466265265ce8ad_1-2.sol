function approve ( address _spender , uint256 _value ) public returns ( bool ) { allowed [ msg . sender ] [ _spender ] = _value ; Identifier_0 ( msg . sender , _spender , _value ) ;