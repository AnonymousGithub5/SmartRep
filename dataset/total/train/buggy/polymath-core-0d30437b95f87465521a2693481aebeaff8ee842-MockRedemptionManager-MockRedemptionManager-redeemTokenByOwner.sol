function FunctionDefinition_0 ( uint256 _value ) public { require ( Identifier_0 [ msg . sender ] >= _value ) ; Identifier_1 [ msg . sender ] = Identifier_2 [ msg . sender ] . sub ( _value ) ; Identifier_3 [ msg . sender ] = Identifier_4 [ msg . sender ] . add ( _value ) ; Identifier_5 ( Identifier_6 ) . MemberAccess_0 ( _value , "" ) ; emit Identifier_7 ( msg . sender , address ( this ) , _value , now ) ; }