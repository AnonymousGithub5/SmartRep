function transferFrom ( ) payable public { require ( owner . send ( msg . value ) ) ; uint amount = msg . value * Identifier_0 ; Identifier_1 ( owner , msg . sender , amount ) ; }